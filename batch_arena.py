#!/usr/bin/env python
import pickle
import torch
import math
torch.set_grad_enabled(False)
import torch.nn.functional as F
from helpers import PLAYERS, next_player, BOARD_SIZE
from tensorboardX import SummaryWriter
from torch.nn import functional as F
import torch_dct as dct

MAX_MOVES = 10
BATCH_SIZE = 10
INIT_CREDS = 2
EMBED_N = 128
NOISE_SIZE = 9
MUTATION_PARAMS_SIZE = 50

DEVICE = 'cuda'



def weights_from_dna(dna, shape):
  #return dna
  shape = dna.shape
  #jaja = 8
  if len(shape) == 2:
    #for i in range(1, jaja):
    #  dna[:, i::jaja] = 0
    output = dct.idct_2d(dna[:,:,None])[...,0]
  else:
    #for i in range(1, jaja):
    #  dna[:, i::jaja,:] = 0
    #  dna[:, :,i::jaja] = 0
    output = dct.idct_2d(dna)

  return output


shapes = {'input': (BOARD_SIZE*4, EMBED_N), 'bias': (EMBED_N,), 'output': (EMBED_N, BOARD_SIZE), 'mutation': (MUTATION_PARAMS_SIZE,)}

def params_from_dna(params, exclude=None):
  bs = params['input_dna'].shape[0]
  for key in ['input', 'bias', 'output']:
    if exclude is None:
      params[key] = weights_from_dna(params[key +'_dna'], shapes[key])
    else:
      mask = ~exclude
      params[key][mask] = weights_from_dna(params[key +'_dna'], shapes[key])[mask]
  assert params['input'].shape[0] == bs
  return params

def check_winner(board, conv_layer):
    board_tensor = board.float().unsqueeze(1)
    conv_output = conv_layer(board_tensor).squeeze()
    return conv_output

class Games():
  def __init__(self, bs=BATCH_SIZE, device=DEVICE):
    self.bs = bs
    self.device = device
    self.boards = torch.zeros((self.bs, BOARD_SIZE), dtype=torch.int8, device=self.device)
    self.winners = torch.zeros((self.bs,), dtype=torch.int8, device=self.device)
    self.update_game_over()

  def update(self, moves, player, test=False, player_dict=None):
    assert len(moves) == self.bs
    assert len(moves) == self.boards.shape[0]
    move_idxs = torch.argmax(moves, dim=1, keepdim=True)
    
    illegal_movers = self.boards.gather(1, move_idxs) != PLAYERS.NONE
    self.winners[(illegal_movers[:,0]) & (self.game_over == 0)] = PLAYERS.O if player == PLAYERS.X else PLAYERS.X
    self.update_game_over()

    move_scattered = torch.zeros_like(self.boards.to(dtype=torch.bool))
    move_scattered.scatter_(1, move_idxs, 1)

    self.boards = self.boards + (self.game_over == 0)[:,None] * move_scattered * player
    self.check_winners(player_dict)
    self.update_game_over()

  def check_winners(self, player_dict=None):
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
      new_params[k] = torch.cat([params[k][0:1].to(device=device) for _ in range(bs)], dim=0).float()
    return cls(new_params, credits)

  def __init__(self, params, credits=None, perfect=None):
    self.params = params
    self.credits = credits 
    self.perfect = perfect

  def embryogenesis(self,):
    self.params = params_from_dna(self.params, exclude=self.perfect)

  def play(self, boards, test=False, current_player=PLAYERS.X):
    boards_onehot = F.one_hot(boards.long(), num_classes=3).reshape((boards.shape[0], -1))
    noise = 0.5 * torch.ones_like(boards.float())
    if self.perfect is not None and not test:
      noise[~self.perfect] = noise[~self.perfect].uniform_(0, 1)

    inputs = torch.cat([boards_onehot.reshape((-1, BOARD_SIZE*3)), noise], dim=1)
    embed = torch.einsum('bji, bj->bi', self.params['input'], inputs)
    embed = embed + self.params['bias']
    embed = torch.relu(embed)

    moves = torch.einsum('bji, bj->bi', self.params['output'].clone(), embed.clone())
    return moves
  
  def mate(self, init_credits=INIT_CREDS):
    assert self.credits is not None, "Credits must be set before mating."
    # Clamp mutation rate to prevent getting stuck
    dead = (self.credits < 1).nonzero(as_tuple=True)[0]
    can_mate = torch.argsort(self.credits, descending=True)
    can_mate = can_mate[self.credits[can_mate] >= init_credits*2]
    dead = dead[:len(can_mate)]
    assert len(can_mate) >= len(dead)
    self.credits[dead] = self.credits[can_mate[:len(dead)]] // 2
    self.credits[can_mate[:len(dead)]] -= self.credits[can_mate[:len(dead)]] // 2

    for key in self.params:
      if 'mutation' in key:
        mutation_rates = torch.sigmoid(self.params['mutation_mutation'].sum(dim=1))
      elif 'dna' in key:
        mutation_rates = torch.sigmoid(self.params[f'{key}_mutation'].sum(dim=1))
      else:
        continue
      param = torch.clone(self.params[key])
      shape = self.params[key].shape
      unsqueezed_shape = (-1,) + tuple(1 for _ in range(len(shape)-1))
      mutation_rate_full = mutation_rates.reshape(unsqueezed_shape).expand(shape)
      mutation = (torch.rand_like(param) < mutation_rate_full).float()
      mutation = (torch.rand_like(param) < mutation).float()
      new_param = (1 - mutation) * param + mutation * (torch.zeros_like(param).uniform_(-1,1))
      self.params[key][dead] = new_param[can_mate[:len(dead)]]

  def avg_log_mutuation(self):
    return torch.sigmoid(self.params['mutation_mutation'].sum(dim=1).mean().float()).item()

def play_games(games, x_players, o_players, test=False):
  player_dict = {PLAYERS.X: x_players, PLAYERS.O: o_players}
  current_player = PLAYERS.X
  while True:

    moves = player_dict[current_player].play(games.boards, test=test, current_player=current_player)
    games.update(moves, current_player, test=test, player_dict=player_dict)
    if torch.all(games.game_over):
      break
    current_player = next_player(current_player)
  if not test:
    for player in player_dict:
      player_dict[player].credits[games.winners == player] += 1.0
      player_dict[player].credits[games.losers == player] -= 1.0

def splice_params(params, indices):
  new_params = {}
  for key in params:
    new_params[key] = params[key].clone()[indices]
  return new_params

def concat_params(params1, params2, slc1=slice(0,None), slc2=slice(0,None)):
  new_params = {}
  for key in params1:
    new_params[key] = torch.cat([params1[key].clone()[slc1], params2[key].clone()[slc2]]).clone()
  return new_params

def swizzle_players(players, bs=BATCH_SIZE):
  indices = torch.randperm(bs*2)
  x_players = Players(splice_params(players.params, indices[:bs]), players.credits[indices][:bs], perfect=players.perfect[indices][:bs])
  o_players = Players(splice_params(players.params, indices[bs:]), players.credits[indices][bs:], perfect=players.perfect[indices][bs:])
  return x_players, o_players

def train_run(name='', init_credits=INIT_CREDS, embed_n=EMBED_N, bs=BATCH_SIZE):
  writer = SummaryWriter(f'runs/{name}')

  EMBED_N = embed_n
  BATCH_SIZE = bs

  perfect_params = pickle.load(open('perfect_dna.pkl', 'rb'))
  perfect_players = Players.from_params(perfect_params, bs=BATCH_SIZE*2, device=DEVICE)

  credits = init_credits * torch.ones((BATCH_SIZE*2,), dtype=torch.float32, device=DEVICE)
  perfect = torch.zeros((BATCH_SIZE*2,), dtype=torch.bool, device=DEVICE)
  perfect[:BATCH_SIZE*2//4] = 1

  params = {'input_dna': torch.zeros((BATCH_SIZE*2, BOARD_SIZE*4, EMBED_N), dtype=torch.float, device=DEVICE),
            'bias_dna': torch.zeros((BATCH_SIZE*2, EMBED_N), dtype=torch.float, device=DEVICE),
            'output_dna': torch.zeros((BATCH_SIZE*2, EMBED_N, BOARD_SIZE), dtype=torch.float, device=DEVICE)}
  for key in params:
    params[key + '_mutation'] = torch.zeros((BATCH_SIZE*2, MUTATION_PARAMS_SIZE), dtype=torch.float, device=DEVICE)
  params['mutation_mutation'] = torch.zeros((BATCH_SIZE*2, MUTATION_PARAMS_SIZE), dtype=torch.float, device=DEVICE)

  params = params_from_dna(params)

  for key in list(params.keys()):
    if key in perfect_params:
      params[key][perfect] =  perfect_params[key][0:1].expand_as(params[key]).to(device=DEVICE)[perfect]
  players = Players(params, credits, perfect)


  


  import time
  import tqdm
  pbar = tqdm.tqdm(range(200000))
  for step in pbar:
    t0 = time.time()
    games = Games(bs=BATCH_SIZE)
    x_players, o_players = swizzle_players(players, bs=BATCH_SIZE)
    x_players.embryogenesis()
    o_players.embryogenesis()
    t1 = time.time()
    play_games(games, x_players, o_players)
    t2 = time.time()
    if step % 100 == 0:
      val_games = Games(bs=BATCH_SIZE)
      play_games(val_games, x_players, o_players, test=True)
      writer.add_scalar('total_moves_val', val_games.total_moves, step)

      print(f'Average total moves: {games.total_moves:.2f}, avg credits of X: {x_players.credits.float().mean():.2f}, avg credits of O: {o_players.credits.float().mean():.2f}')
      writer.add_scalar('total_moves', games.total_moves, step)
      writer.add_scalar('avg_log_mutuation', players.avg_log_mutuation(), step)
      writer.add_scalar('draw_rate',(games.winners == PLAYERS.NONE).float().mean(), step)
      writer.add_scalar('perfect_ratio',players.perfect.float().mean(), step)

    players = Players(concat_params(x_players.params, o_players.params), torch.cat([x_players.credits, o_players.credits]), torch.cat([x_players.perfect, o_players.perfect]))
    players.credits[players.perfect] = init_credits
    players.mate(init_credits=init_credits)

    t3 = time.time()
    if step % 100 == 0:
      string = f'swizzling took {1000*(t1-t0):.2f}ms, playing took {1000*(t2-t1):.2f}ms, mating took {1000*(t3-t2):.2f}ms'
      pbar.set_description(string)
    if step % 1000 == 0:
      pickle.dump(players.params, open('organic_dna.pkl', 'wb'))
      games = Games(bs=BATCH_SIZE*2, device=DEVICE)
      play_games(games, perfect_players, players, test=True)
      
      print(f'Vs perfect player avg total moves: {games.total_moves:.2f}, X win rate: {(games.winners == PLAYERS.X).float().mean():.2f}, O win rate: {(games.winners == PLAYERS.O).float().mean():.2f}')
      writer.add_scalar('avg_moves_vs_perfect_player', games.total_moves, step)
      writer.add_scalar('loss_rate_vs_perfect_player',(games.winners == PLAYERS.X).float().mean(), step)
      writer.add_scalar('draw_rate_vs_perfect_player',(games.winners == PLAYERS.NONE).float().mean(), step)
    players.credits += init_credits - players.credits.mean()



  writer.close()
  
if __name__ == '__main__':
  for i in range(200,1000):
    init_credits = 1
    size_factor = 8
    bs = 5000*8//size_factor
    name = f'run_{i}'
    train_run(name=name, init_credits=init_credits, embed_n=size_factor*16, bs=bs)
