#!/usr/bin/env python
import pickle
import torch
torch.set_grad_enabled(False)
import time
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
    return moves
  
  def mate(self, init_credits=INIT_CREDS):
    assert self.credits is not None, "Credits must be set before mating."
    # Clamp mutation rate to prevent getting stuck
    log_mutation_rate = torch.clamp(self.params['mutuation'].sum(dim=1), -15, 0)
    mutation_rate = torch.exp(log_mutation_rate)
    dead = (self.credits == 0).nonzero(as_tuple=True)[0]
    can_mate = (self.credits > init_credits).nonzero(as_tuple=True)[0]
    assert len(can_mate) >= len(dead)
    self.credits[dead] = init_credits
    self.credits[can_mate[:len(dead)]] -= init_credits
    for key in self.params:
      # repeat mutation rate to match shape of params
      shape = self.params[key].shape
      unsqueezed_shape = (-1,) + tuple(1 for _ in range(len(shape)-1))
      mutation_rate_full = mutation_rate.reshape(unsqueezed_shape).expand(shape)
      param = torch.clone(self.params[key])
      mutation = (torch.rand_like(param) < mutation_rate_full).float()
      new_param = (1 - mutation) * param + mutation * (torch.randn_like(param))
      self.params[key][dead] = new_param[can_mate[:len(dead)]]

  def avg_log_mutuation(self):
    return self.params['mutuation'].sum(dim=1).float().mean().item()

def finish_games(games, x_players, o_players, test=False):
  while True:
    moves = x_players.play(games.boards, test=test)
    games.update(moves, PLAYERS.X)
    if torch.all(games.game_over):
      break
    moves = o_players.play(games.boards, test=test)
    games.update(moves, PLAYERS.O)
    if torch.all(games.game_over):
      break
  if not test:
    x_players.credits[games.winners == PLAYERS.X] += 1
    x_players.credits[games.losers == PLAYERS.X] -= 1
    o_players.credits[games.winners == PLAYERS.O] += 1
    o_players.credits[games.losers == PLAYERS.O] -= 1


def splice_params(params, indices):
  new_params = {}
  for key in params:
    new_params[key] = params[key][indices]
  return new_params

def concat_params(params1, params2):
  new_params = {}
  for key in params1:
    new_params[key] = torch.cat([params1[key], params2[key]])
  return new_params


def train_run(name='', credits=INIT_CREDS):
  writer = SummaryWriter(f'runs/{name}')

  credits = credits * torch.ones((BATCH_SIZE*2,), dtype=torch.int8, device=DEVICE)
  params = {'input': torch.randn((BATCH_SIZE*2, BOARD_SIZE*4, EMBED_N), dtype=torch.float32, device=DEVICE),
            'bias': torch.randn((BATCH_SIZE*2, EMBED_N), dtype=torch.float32, device=DEVICE),
            'output': torch.randn((BATCH_SIZE*2, EMBED_N, BOARD_SIZE), dtype=torch.float32, device=DEVICE),
            'mutuation': torch.randn((BATCH_SIZE*2, MUTATION_PARAMS_SIZE), dtype=torch.float32, device=DEVICE)}
  players = Players(params, credits)

  perfect_params = pickle.load(open('perfect_dna.pkl', 'rb'))
  perfect_players = Players.from_params(perfect_params, bs=BATCH_SIZE*2, device=DEVICE)

  t_start = time.time()
  for step in range(100000):
    t0 = time.time()
    games = Games()
    indices = torch.randperm(BATCH_SIZE*2)
    x_players = Players(splice_params(players.params, indices[:BATCH_SIZE]), players.credits[indices][:BATCH_SIZE])
    o_players = Players(splice_params(players.params, indices[BATCH_SIZE:]), players.credits[indices][BATCH_SIZE:])
    t1 = time.time()
    finish_games(games, x_players, o_players)
    t2 = time.time()
    players = Players(concat_params(x_players.params, o_players.params), torch.cat([x_players.credits, o_players.credits]))
    players.mate()
    t3 = time.time()
    if step % 100 == 0:
      print(f'{step} games took {t3-t_start:.2f} seconds')
      print(t1-t0, t2-t1, t3-t2)
      print(f'Average total moves: {games.total_moves:.2f}, avg credits of X: {x_players.credits.float().mean():.2f}, avg credits of O: {o_players.credits.float().mean():.2f}')
      print(f'Average log mutuation: {players.avg_log_mutuation():.2e}')
      writer.add_scalar('total_moves', games.total_moves, step)
      writer.add_scalar('avg_log_mutuation', players.avg_log_mutuation(), step)
    if step % 1000 == 0:
      pickle.dump(players.params, open('organic_dna.pkl', 'wb'))
      games = Games(bs=BATCH_SIZE*2, device=DEVICE)
      finish_games(games, perfect_players, players, test=True) 
      
      print(f'Vs perfect player avg total moves: {games.total_moves:.2f}, X win rate: {(games.winners == PLAYERS.X).float().mean():.2f}, O win rate: {(games.winners == PLAYERS.O).float().mean():.2f}')
      writer.add_scalar('perfect_total_moves', games.total_moves, step)
      writer.add_scalar('perfect_loss_rate',(games.winners == PLAYERS.X).float().mean(), step)
      writer.add_scalar('perfect_draw_rate',(games.winners == PLAYERS.NONE).float().mean(), step)

  writer.close()
  
if __name__ == '__main__':
  with torch.no_grad():
    for i in range(3,20, 4):
      mutation_rate = 10**(-i)
      name = f'run2_{i}'
      train_run(name=name, credits=i)
