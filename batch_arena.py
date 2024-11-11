#!/usr/bin/env python
import pickle
import torch
import math
torch.set_grad_enabled(False)
import torch.nn.functional as F
from helpers import PLAYERS, next_player, BOARD_SIZE
from tensorboardX import SummaryWriter
from torch.nn import functional as F
#import torch_dct as dct

MAX_MOVES = 10
BATCH_SIZE = 10
INIT_CREDS = 2
EMBED_N = 128
NOISE_SIZE = 16
MUTATION_PARAMS_SIZE = 50

META_INPUT = 8
META_OUTPUT = 3
META_CORE = 32
META_A = META_INPUT*META_CORE
META_B = META_CORE*META_OUTPUT
META_BIAS = META_CORE
META_SIZE = META_A + META_B + META_BIAS

DEVICE = 'cuda'



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
  def from_params(cls, params, bs=1, device=DEVICE):
    new_params = {}
    for k in params:
      new_params[k] = torch.cat([params[k][0:1].to(device=device) for _ in range(bs)], dim=0).float()
    return cls(new_params)

  def __init__(self, params):
    self.bs = params['input'].shape[0]
    self.device = params['input'].device
    self.params = params
    self.reset_state()

  def reset_state(self):
    self.state = torch.zeros((self.bs, NOISE_SIZE), device=self.device)

  def embryogenesis(self):
    matrix_A = self.params['embryogenesis'][:, :META_A].reshape((self.bs, META_INPUT, META_CORE))
    matrix_B = self.params['embryogenesis'][:, META_A:META_A+META_B].reshape((self.bs, META_CORE, META_OUTPUT))
    bias = self.params['embryogenesis'][:, META_A+META_B:META_A+META_B+META_BIAS]



    for key in list(self.params.keys()):
      if 'meta' in key and 'mutation' not in key and 'embryogenesis' not in key:

        #print(key, self.params[key].shape, self.params[key.replace('_meta', '')].shape)
        #self.params[key.replace('_meta', '')] = self.params[key][:,:,0]
        #continue


        embed = torch.einsum('bji, bnj->bni', matrix_A, self.params[key])
        embed = embed + bias[:,None,:]
        embed = torch.relu(embed)
        embed = torch.einsum('bji, bnj->bni', matrix_B, embed)
        self.params[key.replace('_meta', '')] = embed[:,:,0]
        self.params[key + '_mutation'] = embed[:,:,1]
        self.params[key + '_mutation_size'] = embed[:,:,2]

  def play(self, boards, test=False, current_player=PLAYERS.X):
    boards_onehot = F.one_hot(boards.long(), num_classes=3).reshape((boards.shape[0], -1))
    noise = 0.5 * torch.ones((boards.shape[0], NOISE_SIZE), device=boards.device)
    noise = noise.uniform_(0, 1)

    inputs = torch.cat([boards_onehot.reshape((-1, BOARD_SIZE*3)), self.state, noise], dim=1)
    matrix_A = self.params['input'].reshape((self.bs, BOARD_SIZE*3 + NOISE_SIZE*2, EMBED_N))
    matrix_B = self.params['output'].reshape((self.bs,  EMBED_N, BOARD_SIZE + NOISE_SIZE))
    embed = torch.einsum('bji, bj->bi', matrix_A, inputs)
    embed = embed + self.params['bias']
    embed = torch.relu(embed)

    out = torch.einsum('bji, bj->bi', matrix_B, embed)
    moves =  out[:,:BOARD_SIZE]
    #self.state = out[:,BOARD_SIZE:]
    return moves

  def avg_log_mutuation(self):
    return torch.sigmoid(self.params['mutation'].sum(dim=1)).mean().float().item()

  def avg_log_mutuation_size(self):
    return torch.sigmoid(-abs(self.params['embryogenesis_mutation'].sum(dim=1)/10)).mean().float().item()

def mate(x_players, o_players, x_winners, o_winners):
    x_players.embryogenesis()
    o_players.embryogenesis()
    #assert (x_winners != o_winners).all()
    for players, other_players, winners in [(x_players, o_players, x_winners), (o_players, x_players, o_winners)]:
      for key in players.params:
        if 'meta' not in key and 'mutation' not in key and 'embryogenesis' not in key: continue
        if 'meta_mutation' in key: continue
        #if 'meta' in key: continue
        #if 'embryogenesis' in key: continue
        param = players.params[key]
        shape = players.params[key].shape


        if 'meta' in key:
          mutation_logit = -abs(players.params[key+'_mutation'][:,:,None])
        elif len(shape) == 2:
          mutation_logit = -abs(players.params['embryogenesis_mutation']).sum(dim=1)[:,None]/10
        else:
          mutation_logit = -abs(players.params['embryogenesis_mutation']).sum(dim=1)[:,None,None]/10

        unsqueezed_shape = (-1,) + tuple(1 for _ in range(len(shape)-1))
         # mutation_full_logit = mutation_logit.reshape(unsqueezed_shape).expand(shape)
        #if '_std' not in key:
        #  mutation_full_logit = mutation_full_logit +  5 * players.params[key + '_std']
        mutation_rate_full = torch.sigmoid(mutation_logit)
        #print(shape, mutation_rate_full.shape)
        #mutation_full = (mutation_rate_full < torch.rand_like(mutation_rate_full)).float()
        mutation_full = (mutation_rate_full < torch.rand(shape, device=param.device)).float()
        mutation_full2 = (mutation_rate_full < torch.rand(shape, device=param.device)).float()

        #mutation_full = mutation_rate_full
        #mutation_full2 = mutation_rate_full

        other_players.params[key][winners] = param[winners].clone()
        #players.params[key][winners] = (1 - mutation_full[winners]) * players.params[key][winners] + mutation_full[winners] * (torch.zeros_like(players.params[key]).uniform_(-1,1))[winners]
        players.params[key][winners] = (mutation_full * players.params[key] + (mutation_full * (torch.zeros_like(players.params[key]).uniform_(-1,1))))[winners]
        other_players.params[key][winners] = (mutation_full2 * players.params[key] + (mutation_full2 * (torch.zeros_like(other_players.params[key]).uniform_(-1,1))))[winners]
        #players.params[key][winners] += mutation_full[winners] * (torch.zeros_like(players.params[key]).uniform_(-1,1))[winners]
        #other_players.params[key][winners] += mutation_full2[winners] * (torch.zeros_like(other_players.params[key]).uniform_(-1,1))[winners]
    x_players.embryogenesis()
    o_players.embryogenesis()

def play_games(games, x_players, o_players, test=False):
  player_dict = {PLAYERS.X: x_players, PLAYERS.O: o_players}
  current_player = PLAYERS.X
  while True:

    moves = player_dict[current_player].play(games.boards, test=test, current_player=current_player)
    games.update(moves, current_player, test=test, player_dict=player_dict)
    if torch.all(games.game_over):
      break
    current_player = next_player(current_player)

def splice_params(params, indices):
  new_params = {}
  for key in params:
    new_params[key] = params[key][indices]
  return new_params

def concat_params(params1, params2, slc1=slice(0,None), slc2=slice(0,None)):
  new_params = {}
  for key in params1:
    new_params[key] = torch.cat([params1[key][slc1], params2[key][slc2]])
  return new_params

def swizzle_players(players, bs=BATCH_SIZE):
  indices = torch.randperm(bs*2)
  x_players = Players(splice_params(players.params, indices[:bs]))
  o_players = Players(splice_params(players.params, indices[bs:]))
  return x_players, o_players

def train_run(name='', embed_n=EMBED_N, bs=BATCH_SIZE):
  writer = SummaryWriter(f'runs/{name}')

  EMBED_N = embed_n
  BATCH_SIZE = bs

  params = {'input': torch.zeros((BATCH_SIZE*2, (BOARD_SIZE*3 + NOISE_SIZE*2) * EMBED_N), dtype=torch.float, device=DEVICE),
            'bias': torch.zeros((BATCH_SIZE*2, EMBED_N), dtype=torch.float, device=DEVICE),
            'output': torch.zeros((BATCH_SIZE*2, EMBED_N *(BOARD_SIZE + NOISE_SIZE)), dtype=torch.float, device=DEVICE)}
  for key in list(params.keys()):
    meta_shape = tuple(list(params[key].shape) + [META_INPUT])
    params[key + '_meta'] = torch.zeros(meta_shape, dtype=torch.float, device=DEVICE).uniform_(-1, 1)
  params['embryogenesis'] = torch.zeros((BATCH_SIZE*2, META_SIZE), dtype=torch.float, device=DEVICE).uniform_(-1,1)
  params['embryogenesis_mutation'] = torch.zeros((BATCH_SIZE*2, MUTATION_PARAMS_SIZE), dtype=torch.float, device=DEVICE).uniform_(-1,1)
  
  


  import time
  import tqdm
  pbar = tqdm.tqdm(range(200000))
  a_players, b_players = swizzle_players(Players(params), bs=BATCH_SIZE)

  for step in pbar:
    t0 = time.time()
    t1 = time.time()
    a_wins = torch.zeros((BATCH_SIZE,), dtype=torch.int8, device=DEVICE)
    b_wins = torch.zeros((BATCH_SIZE,), dtype=torch.int8, device=DEVICE)
    for i in range(1):
      a_players.reset_state()
      b_players.reset_state()
      if i % 2 == 0:
        games = Games(bs=BATCH_SIZE)
        play_games(games, a_players, b_players)
        a_wins += (games.winners == PLAYERS.X)
        b_wins += (games.winners == PLAYERS.O)
      else:
        games = Games(bs=BATCH_SIZE)
        play_games(games, b_players, a_players)
        a_wins += (games.winners == PLAYERS.O)
        b_wins += (games.winners == PLAYERS.X)
    #mut_rate = a_players.avg_log_mutuation()
    mut_size = a_players.avg_log_mutuation_size()
    t2 = time.time()
    mate(a_players, b_players, a_wins > b_wins, a_wins < b_wins)
    t3 = time.time()
    a_players, b_players = swizzle_players(Players(concat_params(a_players.params, b_players.params)), bs=BATCH_SIZE)
    t4 = time.time()
    if step % 100 == 0:
      writer.add_scalar('total_moves_val', games.total_moves, step)

      print(f'Average total moves: {games.total_moves:.2f}, Average mutuation rate: {mut_size:.1e}')
      writer.add_scalar('total_moves', games.total_moves, step)
      #writer.add_scalar('avg_log_mutuation', mut_rate, step)
      writer.add_scalar('avg_log_mutuation_size', mut_size, step)
      writer.add_scalar('draw_rate',(a_wins == b_wins).float().mean(), step)

    if step % 100 == 0:
      string = f'swizzling took {1000*(t4-t3):.2f}ms, playing took {1000*(t2-t1):.2f}ms, mating took {1000*(t3-t2):.2f}ms'
      pbar.set_description(string)
    if step % 1000 == 0:
      pickle.dump(a_players.params, open('organic_dna.pkl', 'wb'))


  writer.close()
  
if __name__ == '__main__':
  for i in range(2000,13000):
    bs = 1000
    name = f'run_{i}'
    train_run(name=name, embed_n=EMBED_N, bs=bs)
