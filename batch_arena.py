#!/usr/bin/env python
import pickle
import torch
torch.set_grad_enabled(False)
import torch.nn.functional as F
from helpers import PLAYERS, next_player, BOARD_SIZE
from tensorboardX import SummaryWriter


MAX_MOVES = 10
BATCH_SIZE = 10000
INIT_CREDS = 2
EMBED_N = 128
NOISE_SIZE = 9
MUTATION_PARAMS_SIZE = 50

DEVICE = 'cuda'


class Games():
  def __init__(self, bs=BATCH_SIZE, device=DEVICE):
    self.bs = bs
    self.device = device
    self.boards = torch.zeros((self.bs, BOARD_SIZE), dtype=torch.int8, device=self.device)
    self.winners = torch.zeros((self.bs,), dtype=torch.int8, device=self.device)
    self.update_game_over()

  def update(self, moves, player):
    assert len(moves) == self.bs
    assert len(moves) == self.boards.shape[0]

    move_idxs = torch.argmax(moves, dim=1, keepdim=True)
    
    illegal_movers = self.boards.gather(1, move_idxs) != PLAYERS.NONE
    self.winners[(illegal_movers[:,0]) & (self.game_over == 0)] = PLAYERS.O if player == PLAYERS.X else PLAYERS.X
    self.update_game_over()

    move_scattered = torch.zeros_like(self.boards.to(dtype=torch.bool))
    move_scattered.scatter_(1, move_idxs, 1)

    self.boards = self.boards + (self.game_over == 0)[:,None] * move_scattered * player
    self.check_winners()
    self.update_game_over()

  def check_winners(self):
    boards = self.boards.reshape((-1, 3, 3))
    M, rows, cols = boards.shape
    assert rows == 3 and cols == 3, "Each board must be a 3x3 grid."
    winners = torch.zeros(M, dtype=torch.int8, device=self.device)
    for player in [PLAYERS.X, PLAYERS.O]:
      rows_winner = torch.any(torch.all(boards == player, dim=1), dim=1)
      cols_winner = torch.any(torch.all(boards == player, dim=2), dim=1)
      winners[rows_winner | cols_winner] = player

      diag1 = boards[:, torch.arange(3), torch.arange(3)]
      diag2 = boards[:, torch.arange(3), torch.arange(2, -1, -1)]
      diag1_winner = torch.all(diag1 == player, dim=1)
      diag2_winner = torch.all(diag2 == player, dim=1)
      winners[diag1_winner | diag2_winner] = player
    
    self.winners[self.winners == PLAYERS.NONE] = winners[self.winners == PLAYERS.NONE]

  @property
  def losers(self):
    losers = torch.zeros_like(self.winners)
    losers[self.winners == PLAYERS.X] = PLAYERS.O
    losers[self.winners == PLAYERS.O] = PLAYERS.X
    return losers
  
  @property
  def total_moves(self):
    return (self.boards != PLAYERS.NONE).sum(dim=1).float().mean()

  def update_game_over(self):
    self.game_over = (self.winners != PLAYERS.NONE) | ((self.boards != PLAYERS.NONE).sum(dim=1) == BOARD_SIZE)


class Players():

  @classmethod
  def from_params(cls, params, bs=1, credits=None, device=DEVICE):
    new_params = {}
    for k in params:
      new_params[k] =  torch.cat([params[k][0:1].to(device=device)  for _ in range(bs)], dim=0)
    return cls(new_params, credits)

  def __init__(self, params, credits=None):
    self.params = params
    self.credits = credits

  def play(self, boards, test=False):
    boards_onehot = F.one_hot(boards.long(), num_classes=3).reshape((boards.shape[0], -1))
    if test:
      noise = 0.5*torch.ones_like(boards.float())
    else:
      noise = torch.randn_like(boards.float())
    inputs = torch.cat([boards_onehot.reshape((-1, BOARD_SIZE*3)), noise], dim=1)
    embed = torch.einsum('bji, bj->bi', self.params['input'], inputs)
    embed = embed + self.params['bias']
    embed = torch.relu(embed)
    moves = torch.einsum('bji, bj->bi', self.params['output'], embed)
    moves = torch.softmax(moves, dim=1)
    #if not test:
    #  random_moves = torch.rand(moves.shape[:1], device=DEVICE) < 1e-2
    #  moves = (1 - random_moves.float()[:,None]) * moves + random_moves[:,None] * torch.rand_like(moves)
    return moves
  
  def mate(self, init_credits=INIT_CREDS):
    assert self.credits is not None, "Credits must be set before mating."
    # Clamp mutation rate to prevent getting stuck
    log_mutation_rate = torch.clamp(self.params['mutuation'].sum(dim=1), -15, 0)
    mutation_rate = torch.exp(log_mutation_rate)
    dead = (self.credits == 0).nonzero(as_tuple=True)[0]
    can_mate = (self.credits > init_credits).nonzero(as_tuple=True)[0]
    assert len(can_mate) >= len(dead)
    self.credits[dead] = self.credits[can_mate[:len(dead)]] - self.credits[can_mate[:len(dead)]] // 2
    self.credits[can_mate[:len(dead)]] = self.credits[can_mate[:len(dead)]] // 2
    for key in self.params:
      # repeat mutation rate to match shape of params
      shape = self.params[key].shape
      unsqueezed_shape = (-1,) + tuple(1 for _ in range(len(shape)-1))
      mutation_rate_full = mutation_rate.reshape(unsqueezed_shape).expand(shape)
      param = torch.clone(self.params[key])
      mutation = (torch.rand_like(param) < mutation_rate_full).float()
      new_param = (1 - mutation) * param + mutation * (torch.randn_like(param))
      self.params[key][dead] = new_param[can_mate[:len(dead)]]
      mutation2 = (torch.rand_like(param) < mutation_rate_full).float()
      new_param2 = (1 - mutation) * param + mutation2 * (torch.randn_like(param))
      self.params[key][can_mate[:len(dead)]] = new_param2[can_mate[:len(dead)]]

  def avg_log_mutuation(self):
    return self.params['mutuation'].sum(dim=1).float().mean().item()

def play_games(games, x_players, o_players, test=False):
  player_dict = {PLAYERS.X: x_players, PLAYERS.O: o_players}
  current_player = PLAYERS.X
  while True:

    moves = player_dict[current_player].play(games.boards, test=test)
    games.update(moves, current_player)
    if torch.all(games.game_over):
      break
    current_player = next_player(current_player)
  if not test:
    for player in player_dict:
      player_dict[player].credits[games.winners == player] += 1
      player_dict[player].credits[games.losers == player] -= 1

def splice_params(params, indices):
  new_params = {}
  for key in params:
    new_params[key] = params[key][indices]
  return new_params

def concat_params(params1, params2, slc1=slice(0,None), slc2=slice(0,None)):
  new_params = {}
  for key in params1:
    new_params[key] = torch.cat([params1[key][slc1], params2[key][slc2]]).clone()
  return new_params

def swizzle_players(players):
  #n = torch.randint(0, BATCH_SIZE, (1,)).item()
  #x_players = Players(concat_params(players.params, players.params, slc1=slice(n,n+BATCH_SIZE), slc2=slice(n,n)), torch.cat([players.credits[n:n+BATCH_SIZE], players.credits[n:n]]))
  #o_players = Players(concat_params(players.params, players.params, slc1=slice(0, n), slc2=slice(n+BATCH_SIZE,None)), torch.cat([players.credits[:n], players.credits[n+BATCH_SIZE:]]))
  #return x_players, o_players

  indices = torch.randperm(BATCH_SIZE*2)
  x_players = Players(splice_params(players.params, indices[:BATCH_SIZE]), players.credits[indices][:BATCH_SIZE])
  o_players = Players(splice_params(players.params, indices[BATCH_SIZE:]), players.credits[indices][BATCH_SIZE:])
  return x_players, o_players


def train_run(name='', init_credits=INIT_CREDS):
  writer = SummaryWriter(f'runs/{name}')

  credits = init_credits * torch.ones((BATCH_SIZE*2,), dtype=torch.int8, device=DEVICE)
  params = {'input': torch.randn((BATCH_SIZE*2, BOARD_SIZE*4, EMBED_N), dtype=torch.float32, device=DEVICE),
            'bias': torch.randn((BATCH_SIZE*2, EMBED_N), dtype=torch.float32, device=DEVICE),
            'output': torch.randn((BATCH_SIZE*2, EMBED_N, BOARD_SIZE), dtype=torch.float32, device=DEVICE),
            'mutuation': torch.randn((BATCH_SIZE*2, MUTATION_PARAMS_SIZE), dtype=torch.float32, device=DEVICE)}
  players = Players(params, credits)

  perfect_params = pickle.load(open('perfect_dna.pkl', 'rb'))
  perfect_players = Players.from_params(perfect_params, bs=BATCH_SIZE*2, device=DEVICE)

  import time
  import tqdm
  pbar = tqdm.tqdm(range(500000))
  for step in pbar:
    t0 = time.time()
    games = Games()
    x_players, o_players = swizzle_players(players)
    t1 = time.time()
    play_games(games, x_players, o_players)
    t2 = time.time()
    players = Players(concat_params(x_players.params, o_players.params), torch.cat([x_players.credits, o_players.credits]))
    players.mate(init_credits=init_credits)
    t3 = time.time()
    if step % 100 == 0:
      print(f'Average total moves: {games.total_moves:.2f}, avg credits of X: {x_players.credits.float().mean():.2f}, avg credits of O: {o_players.credits.float().mean():.2f}')
      writer.add_scalar('total_moves', games.total_moves, step)
      writer.add_scalar('avg_log_mutuation', players.avg_log_mutuation(), step)
      string = f'swizzling took {1000*(t1-t0):.2f}ms, playing took {1000*(t2-t1):.2f}ms, mating took {1000*(t3-t2):.2f}ms'
      pbar.set_description(string)
    if step % 1000 == 0:
      pickle.dump(players.params, open('organic_dna.pkl', 'wb'))
      games = Games(bs=BATCH_SIZE*2, device=DEVICE)
      play_games(games, perfect_players, players, test=True) 
      
      #print(f'Vs perfect player avg total moves: {games.total_moves:.2f}, X win rate: {(games.winners == PLAYERS.X).float().mean():.2f}, O win rate: {(games.winners == PLAYERS.O).float().mean():.2f}')
      writer.add_scalar('perfect_total_moves', games.total_moves, step)
      writer.add_scalar('perfect_loss_rate',(games.winners == PLAYERS.X).float().mean(), step)
      writer.add_scalar('perfect_draw_rate',(games.winners == PLAYERS.NONE).float().mean(), step)

  writer.close()
  
if __name__ == '__main__':
  for i in [16]:
    mutation_rate = 10**(-i)
    name = f'run2_{i}'
    train_run(name=name, init_credits=3)
