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
META_CORE = 16
META_SIZE = META_CORE*3 + 1

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
  bs = params['input'].shape[0]
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
    self.mutation_rate = torch.zeros_like(self.credits)
    self.reset_state()

  def reset_state(self):
    self.state = torch.zeros((self.credits.shape[0], NOISE_SIZE), device=self.credits.device)





  def play(self, boards, test=False, current_player=PLAYERS.X):
    boards_onehot = F.one_hot(boards.long(), num_classes=3).reshape((boards.shape[0], -1))
    noise = 0.5 * torch.ones((boards.shape[0], NOISE_SIZE), device=boards.device)
    noise = noise.uniform_(0, 1)
    #if self.perfect is not None and not test:
    #  noise[~self.perfect] = noise[~self.perfect].uniform_(0, 1)

    inputs = torch.cat([boards_onehot.reshape((-1, BOARD_SIZE*3)), self.state, noise], dim=1)
    embed = torch.einsum('bji, bj->bi', self.params['input'], inputs)
    embed = embed + self.params['bias']
    embed = torch.relu(embed)

    #res_x = torch.einsum('bji, bj->bi', self.params['res_a'], embed)
    #res_x = torch.einsum('bji, bj->bi', self.params['res_b'], torch.relu(res_x + self.params['res_bias']))

    out = torch.einsum('bji, bj->bi', self.params['output'], embed)# + res_x)
    moves =  out[:,:BOARD_SIZE]
    #self.state = out[:,BOARD_SIZE:]
    return moves
  
  def embryogenesis(self):
    for key in self.params:
      if 'meta' in key:
        dna_key = key.replace('_meta', '')
        orig_size = self.params[dna_key].shape
        meta_input = self.params[key].reshape((-1, META_SIZE))[:,:META_CORE].reshape((-1, 1, META_CORE))
        meta_bias = self.params[key].reshape((-1, META_SIZE))[:,META_CORE:META_CORE*2].reshape((-1, META_CORE))
        meta_output = self.params[key].reshape((-1, META_SIZE))[:,META_CORE*2:META_CORE*3].reshape((-1, META_CORE, 1))
        meta_cst = self.params[key].reshape((-1, META_SIZE))[:,-1].reshape((-1, 1))
        x = torch.einsum('bji, bj->bi', meta_input, meta_cst)
        x = x + meta_bias
        x = torch.relu(x)
        x = torch.einsum('bji, bj->bi', meta_output, x)
        self.params[dna_key] = x.reshape(orig_size)

  @property
  def init_credits(self):
    return 1 + 0*torch.sigmoid(self.params['init_credits'].sum(dim=1))

  def avg_log_mutuation(self):
    return torch.sigmoid(self.params['mutation'].sum(dim=1)).mean().float().item()

  def avg_log_mutuation_size(self):
    return torch.sigmoid(self.params['mutation_size'].sum(dim=1)).mean().float().item()

def mate(x_players, o_players, x_winners, o_winners):
    for players, other_players, winners in [(x_players, o_players, x_winners), (o_players, x_players, o_winners)]:
      for key in players.params:
        param = players.params[key]
        shape = players.params[key].shape
        if '_std' in key:
          #mutation_rates = torch.sigmoid(players.params['mutation'].sum(dim=1))
          mutation_sizes = torch.sigmoid(players.params['mutation_size'].sum(dim=1))
          unsqueezed_shape = (-1,) + tuple(1 for _ in range(len(shape)-1))
          #mutation_rate_full = mutation_rates.reshape(unsqueezed_shape).expand(shape)
          mutation_sizes_full = mutation_sizes.reshape(unsqueezed_shape).expand(shape)
        else:
          mutation_sizes_full = torch.sigmoid(players.params[key + '_std'])

        #mutation = (torch.rand_like(param) < mutation_rate_full).float()
        #mutation2 = (torch.rand_like(param) < mutation_rate_full).float()
        #new_param = (1 - mutation) * param + mutation * (torch.zeros_like(param).uniform_(-1,1))
        #new_param2 = (1 - mutation2) * param + mutation2 * (torch.zeros_like(param).uniform_(-1,1))
        other_players.params[key][winners] = param[winners].clone()
        #other_players.params[key][winners]

        #mutation = (torch.rand_like(param) < mutation_rate_full)
        #mutation2 = (torch.rand_like(param) < mutation_rate_full).float()
        players.params[key] += mutation_sizes_full * (torch.zeros_like(players.params[key]).uniform_(-1,1))
        #other_players.params[key] += mutation2 * other_players.params[key].uniform_(-1,1)
        other_players.params[key] += mutation_sizes_full * (torch.zeros_like(other_players.params[key]).uniform_(-1,1))
        
        #other_players.params[key][winners] = new_param[winners]
        #players.params[key][winners] = new_param2[winners]
    #x_players.embryogenesis()
    #o_players.embryogenesis()

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
    new_params[key] = params[key][indices]
  return new_params

def concat_params(params1, params2, slc1=slice(0,None), slc2=slice(0,None)):
  new_params = {}
  for key in params1:
    new_params[key] = torch.cat([params1[key][slc1], params2[key][slc2]])
  return new_params

def swizzle_players(players, bs=BATCH_SIZE):
  indices = torch.randperm(bs*2)
  x_players = Players(splice_params(players.params, indices[:bs]), players.credits[indices][:bs])
  o_players = Players(splice_params(players.params, indices[bs:]), players.credits[indices][bs:])
  return x_players, o_players

def train_run(name='', embed_n=EMBED_N, bs=BATCH_SIZE):
  init_credits = 1
  writer = SummaryWriter(f'runs/{name}')

  EMBED_N = embed_n
  BATCH_SIZE = bs

  #perfect_params = pickle.load(open('perfect_dna.pkl', 'rb'))
  #perfect_players = Players.from_params(perfect_params, bs=BATCH_SIZE*2, device=DEVICE)

  credits = init_credits * torch.ones((BATCH_SIZE*2,), dtype=torch.float32, device=DEVICE)
  #perfect = torch.zeros((BATCH_SIZE*2,), dtype=torch.bool, device=DEVICE)
  #perfect[:BATCH_SIZE*2//4] = 1

  params = {'input': torch.zeros((BATCH_SIZE*2, BOARD_SIZE*3 + NOISE_SIZE*2, EMBED_N), dtype=torch.float, device=DEVICE),
            'bias': torch.zeros((BATCH_SIZE*2, EMBED_N), dtype=torch.float, device=DEVICE),
            'output': torch.zeros((BATCH_SIZE*2, EMBED_N, BOARD_SIZE + NOISE_SIZE), dtype=torch.float, device=DEVICE),
            #'res_a': torch.zeros((BATCH_SIZE*2, EMBED_N, EMBED_N*8), dtype=torch.float, device=DEVICE),
            #'res_b': torch.zeros((BATCH_SIZE*2, EMBED_N*8, EMBED_N), dtype=torch.float, device=DEVICE),
            #'res_bias': torch.zeros((BATCH_SIZE*2, EMBED_N*8), dtype=torch.float, device=DEVICE),
            'init_credits': torch.zeros((BATCH_SIZE*2, MUTATION_PARAMS_SIZE), dtype=torch.float, device=DEVICE),
            'mutation': torch.zeros((BATCH_SIZE*2, MUTATION_PARAMS_SIZE), dtype=torch.float, device=DEVICE),
            'mutation_size': torch.zeros((BATCH_SIZE*2, MUTATION_PARAMS_SIZE), dtype=torch.float, device=DEVICE)}
  for key in list(params.keys()):
    #params[key] = params[key].uniform_(-1, 1)
    meta_shape = tuple(list(params[key].shape))
    #params[key + '_meta'] = torch.zeros(meta_shape, dtype=torch.float, device=DEVICE)
    params[key + '_std'] = torch.zeros(meta_shape, dtype=torch.float, device=DEVICE)

  #params = params_from_dna(params)

  #for key in list(params.keys()):
  #  if key in perfect_params:
  #    params[key][perfect] =  perfect_params[key][0:1].expand_as(params[key]).to(device=DEVICE)[perfect]
  players = Players(params, credits)


  


  import time
  import tqdm
  pbar = tqdm.tqdm(range(200000))
  a_players, b_players = swizzle_players(players, bs=BATCH_SIZE)

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
    mut_rate = a_players.avg_log_mutuation()
    mut_size = a_players.avg_log_mutuation_size()
    t2 = time.time()
    mate(a_players, b_players, a_wins > b_wins, a_wins < b_wins)
    t3 = time.time()
    a_players, b_players = swizzle_players(Players(concat_params(a_players.params, b_players.params), torch.cat([a_players.credits, b_players.credits])), bs=BATCH_SIZE)
    t4 = time.time()
    if step % 100 == 0:
      #val_games = Games(bs=BATCH_SIZE)
      #play_games(val_games, x_players, o_players)
      writer.add_scalar('total_moves_val', games.total_moves, step)

      print(f'Average total moves: {games.total_moves:.2f}, avg credits of X: {a_players.credits.float().mean():.2f}, avg credits of O: {b_players.credits.float().mean():.2f}')
      writer.add_scalar('total_moves', games.total_moves, step)
      writer.add_scalar('avg_log_mutuation', mut_rate, step)
      writer.add_scalar('avg_log_mutuation_size', mut_size, step)
      writer.add_scalar('avg_init_credits', a_players.init_credits.mean(), step)
      writer.add_scalar('draw_rate',(a_wins == b_wins).float().mean(), step)
      #writer.add_scalar('perfect_ratio',players.perfect.float().mean(), step)

    #players = Players(concat_params(x_players.params, o_players.params), torch.cat([x_players.credits, o_players.credits]))
    #players.credits[players.perfect] = init_credits

    if step % 100 == 0:
      string = f'swizzling took {1000*(t4-t3):.2f}ms, playing took {1000*(t2-t1):.2f}ms, mating took {1000*(t3-t2):.2f}ms'
      pbar.set_description(string)
    #if step % 1000 == 0:
    #  pickle.dump(players.params, open('organic_dna.pkl', 'wb'))
    #  games = Games(bs=BATCH_SIZE*2, device=DEVICE)
    #  play_games(games, perfect_players, players, test=True)
      
    #  print(f'Vs perfect player avg total moves: {games.total_moves:.2f}, X win rate: {(games.winners == PLAYERS.X).float().mean():.2f}, O win rate: {(games.winners == PLAYERS.O).float().mean():.2f}')
    ##  writer.add_scalar('avg_moves_vs_perfect_player', games.total_moves, step)
    #  writer.add_scalar('loss_rate_vs_perfect_player',(games.winners == PLAYERS.X).float().mean(), step)
    #  writer.add_scalar('draw_rate_vs_perfect_player',(games.winners == PLAYERS.NONE).float().mean(), step)
    #players.credits += players.init_credits.mean() - players.credits.mean()



  writer.close()
  
if __name__ == '__main__':
  for i in range(100,2000):
    bs = 10000
    name = f'run_{i}'
    train_run(name=name, embed_n=EMBED_N, bs=bs)
