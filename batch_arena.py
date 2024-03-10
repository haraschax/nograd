#!/usr/bin/env python
import pickle
import torch
torch.set_grad_enabled(False)
import torch.nn.functional as F
from helpers import PLAYERS, next_player, BOARD_SIZE
from tensorboardX import SummaryWriter
from torch.nn import functional as F

MAX_MOVES = 10
BATCH_SIZE = 10
INIT_CREDS = 2
EMBED_N = 128
NOISE_SIZE = 9
MUTATION_PARAMS_SIZE = 500
ALPHA = .3

DEVICE = 'cuda'

UNPACK1 = torch.nn.Sequential(torch.nn.Linear(EMBED_N + BOARD_SIZE*4, EMBED_N * BOARD_SIZE* 4).to(DEVICE).half(),
                              torch.nn.ReLU(),
                              torch.nn.Linear(EMBED_N * BOARD_SIZE* 4, EMBED_N).to(DEVICE).half(),
                              torch.nn.ReLU(),
                              torch.nn.Linear(EMBED_N, EMBED_N * BOARD_SIZE* 4).to(DEVICE).half())
UNPACK2 = torch.nn.Sequential(torch.nn.Linear(EMBED_N + BOARD_SIZE, EMBED_N * BOARD_SIZE).to(DEVICE).half(),
                              torch.nn.ReLU(),
                              torch.nn.Linear(EMBED_N * BOARD_SIZE, EMBED_N).to(DEVICE).half(),
                              torch.nn.ReLU(),
                              torch.nn.Linear(EMBED_N, EMBED_N * BOARD_SIZE).to(DEVICE).half())



def create_tictactoe_conv_layer():
    filters = torch.tensor([
        [[1, 1, 1], [0, 0, 0], [0, 0, 0]],  # Horizontal
        [[0, 0, 0], [1, 1, 1], [0, 0, 0]],  # Horizontal
        [[0, 0, 0], [0, 0, 0], [1, 1, 1]],  # Horizontal
        [[1, 0, 0], [1, 0, 0], [1, 0, 0]],  # Vertical
        [[0, 1, 0], [0, 1, 0], [0, 1, 0]],  # Vertical
        [[0, 0, 1], [0, 0, 1], [0, 0, 1]],  # Vertical
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # Diagonal
        [[0, 0, 1], [0, 1, 0], [1, 0, 0]]   # Diagonal
    ], dtype=torch.float16)

    filters = filters.unsqueeze(1)
    conv_layer = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, bias=False)
    conv_layer.weight.data = filters
    return conv_layer

conv_layer = create_tictactoe_conv_layer().to(DEVICE)


def check_winner(board, conv_layer):
    board_tensor = board.half().unsqueeze(1)
    conv_output = conv_layer(board_tensor).squeeze()
    return conv_output

class Games():
  def __init__(self, bs=BATCH_SIZE, device=DEVICE):
    self.bs = bs
    self.device = device
    self.boards = torch.zeros((self.bs, BOARD_SIZE), dtype=torch.int8, device=self.device)
    self.winners = torch.zeros((self.bs,), dtype=torch.int8, device=self.device)
    self.update_game_over()

  def update(self, moves, player, test=False):
    assert len(moves) == self.bs
    assert len(moves) == self.boards.shape[0]

    if test:
      move_idxs = torch.argmax(moves, dim=1, keepdim=True)
    else:
      move_idxs = torch.multinomial(moves, num_samples=1)
    #move_idxs = torch.argmax(moves, dim=1, keepdim=True)
    
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
    '''

    # The *3 - 2 is a trick to convert 1,2 to 1,4
    def rescale(x):
      return x ** 2

    a = check_winner(rescale(boards), conv_layer)
    for player in [PLAYERS.X, PLAYERS.O]:
      #print(player, player*3 - 2)
      idxs = (a == 3* (rescale(player))).any(dim=1)
      #print(idxs.shape)
      winners[idxs] = player
      #print(winners)
    self.winners[self.winners == PLAYERS.NONE] = winners[self.winners == PLAYERS.NONE]
    '''

  @property
  def losers(self):
    losers = torch.zeros_like(self.winners)
    losers[self.winners == PLAYERS.X] = PLAYERS.O
    losers[self.winners == PLAYERS.O] = PLAYERS.X
    return losers
  
  @property
  def total_moves(self):
    return (self.boards != PLAYERS.NONE).sum(dim=1).half().mean()

  def update_game_over(self):
    self.game_over = (self.winners != PLAYERS.NONE) | ((self.boards != PLAYERS.NONE).sum(dim=1) == BOARD_SIZE)


class Players():

  @classmethod
  def from_params(cls, params, bs=1, credits=None, device=DEVICE):
    new_params = {}
    for k in params:
      new_params[k] = torch.cat([params[k][0:1].to(device=device) for _ in range(bs)], dim=0).half()
    return cls(new_params, credits)

  def __init__(self, params, credits=None):
    if len(params['input'].shape) == 2:
      params['input_w'] = UNPACK1(params['input']).detach().reshape((-1, BOARD_SIZE*4, EMBED_N))
      params['output_w'] = UNPACK2(params['output']).detach().reshape((-1, EMBED_N, BOARD_SIZE))
    else:
      params['input_w'] = params['input']
      params['output_w'] = params['output']
    self.params = params
    self.credits = credits
    self.alpha = 1.0

  def play(self, boards, test=False, err=0.0):
    boards_onehot = F.one_hot(boards.long(), num_classes=3).reshape((boards.shape[0], -1))
    if test:
      noise = 0.5*torch.ones_like(boards.half())
    else:
      noise = torch.rand_like(boards.half())
    noise = 0.5*torch.ones_like(boards.half())

    inputs = torch.cat([boards_onehot.reshape((-1, BOARD_SIZE*3)), noise], dim=1)
    embed = torch.einsum('bji, bj->bi', self.params['input_w'], inputs)
    embed = embed + self.params['bias']
    embed = torch.relu(embed)

    if 'amplitude' in self.params:
      amplitude = self.params['amplitude'].sum(dim=1).reshape((-1,1))
    else:
      amplitude = 1.0
    amplitude = 5.0
    if 'block_a' in self.params:
      for i in range(3):
        embed = torch.einsum('bji, bj->bi', self.params['block_a'][:,i], embed)
        embed = torch.relu(embed + self.params['block_bias_a'][:,i])
        embed = torch.einsum('bji, bj->bi', self.params['block_b'][:,i], embed)
        embed = torch.relu(embed + self.params['block_bias_b'][:,i])
    moves = torch.einsum('bji, bj->bi', self.params['output_w'], embed)
    moves = torch.nn.functional.normalize(moves, dim=1, eps=1e-3)
    moves = torch.clamp(amplitude * moves, -1e3, 1e3)
    moves = torch.softmax(moves, dim=1)
    if not test:
      random_moves = torch.rand(moves.shape[:1], device=DEVICE) < err
      moves = (1 - random_moves.float()[:,None]) * moves + random_moves[:,None] * torch.rand_like(moves)
    return moves
  
  def mate(self, init_credits=INIT_CREDS, alpha=ALPHA):
    self.alpha = alpha
    assert self.credits is not None, "Credits must be set before mating."
    # Clamp mutation rate to prevent getting stuck
    log_mutation_rates = torch.clamp(self.params['mutation'].sum(dim=1) / (alpha * 10), -15, 0)
    mutation_rates = torch.exp(log_mutation_rates)
    dead = (self.credits < 1).nonzero(as_tuple=True)[0]
    can_mate = torch.argsort(self.credits, descending=True)#(self.credits > init_credits).nonzero(as_tuple=True)[0]
    can_mate = can_mate[self.credits[can_mate] >= init_credits*2]
    dead = dead[:len(can_mate)]
    assert len(can_mate) >= len(dead)
    self.credits[dead] += self.credits[can_mate[:len(dead)]] - self.credits[can_mate[:len(dead)]] // 2
    self.credits[can_mate[:len(dead)]] = self.credits[can_mate[:len(dead)]] // 2

    amplitude = torch.clamp(self.params['amplitude'].sum(dim=1)/10, -10, 10)
    amplitude = torch.exp(amplitude).reshape(-1,1)
      
    for key in self.params:
      if '_' in key:
        continue
      # repeat mutation rate to match shape of params
      shape = self.params[key].shape
      unsqueezed_shape = (-1,) + tuple(1 for _ in range(len(shape)-1))
      #mutation_rate_full = torch.exp(self.params[key + '_mutation'])
      mutation_rate_full = mutation_rates.reshape(unsqueezed_shape).expand(shape)
      param = torch.clone(self.params[key])
      mutation = (torch.rand_like(param) < mutation_rate_full).half()
      new_param = (1 - 1*mutation) * param + mutation * (torch.zeros_like(param).uniform_(-alpha,alpha))
      self.params[key][dead] = new_param[can_mate[:len(dead)]]
      mutation2 = (torch.rand_like(param) < mutation_rate_full).half()
      new_param2 = (1 - 1*mutation) * param + mutation2 * (torch.zeros_like(param).uniform_(-alpha,alpha))
      self.params[key][can_mate[:len(dead)]] = new_param2[can_mate[:len(dead)]]
    for key in self.params:
      if False and 'mutation' in key:
        #shape = self.params[key].shape
        #unsqueezed_shape = (-1,) + tuple(1 for _ in range(len(shape)-1))
        #amplitude_full = amplitude.reshape(unsqueezed_shape)
        mutate_change = torch.zeros_like(self.params[key]).uniform_(-alpha,alpha)
        self.params[key][dead] = self.params[key][can_mate[:len(dead)]] + (mutate_change)[can_mate[:len(dead)]]
        mutate_change2 = torch.zeros_like(self.params[key]).uniform_(-alpha,alpha)
        self.params[key][can_mate[:len(dead)]] = self.params[key][can_mate[:len(dead)]] + (mutate_change2)[can_mate[:len(dead)]]
        self.params[key] = torch.clamp(self.params[key], -30, 2)

  def avg_log_mutuation(self):
    return self.params['mutation'].sum(dim=1).mean().half().item()
    #return self.params['input_mutation'].mean().half().item()

  #def avg_log_amplitude(self):
  #  return self.params['amplitude'].sum(dim=1).mean().half().item() / 10

    
def play_games(games, x_players, o_players, test=False, err=0.0):
  player_dict = {PLAYERS.X: x_players, PLAYERS.O: o_players}
  current_player = PLAYERS.X
  while True:

    moves = player_dict[current_player].play(games.boards, test=test, err=err)
    games.update(moves, current_player, test=test)
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
    new_params[key] = torch.cat([params1[key][slc1], params2[key][slc2]]).clone()
  return new_params

def swizzle_players(players, bs=BATCH_SIZE):
  indices = torch.randperm(bs*2)
  x_players = Players(splice_params(players.params, indices[:bs]), players.credits[indices][:bs])
  o_players = Players(splice_params(players.params, indices[bs:]), players.credits[indices][bs:])
  return x_players, o_players


def train_run(name='', init_credits=INIT_CREDS, embed_n=EMBED_N, bs=BATCH_SIZE, alpha=ALPHA, err=0.1):
  writer = SummaryWriter(f'runs/{name}')

  COMPRESS = 8
  EMBED_N = embed_n
  BATCH_SIZE = bs
  credits = init_credits * torch.ones((BATCH_SIZE*2,), dtype=torch.float32, device=DEVICE)
  #params = {'input': torch.zeros((BATCH_SIZE*2, BOARD_SIZE*4, EMBED_N), dtype=torch.float16, device=DEVICE),
  #          'bias': torch.zeros((BATCH_SIZE*2, EMBED_N), dtype=torch.float16, device=DEVICE),
  #          #'block_a': torch.randn((BATCH_SIZE*2, 3, EMBED_N, EMBED_N*COMPRESS), dtype=torch.float16, device=DEVICE),
  #          #'block_bias_a': torch.randn((BATCH_SIZE*2, 3, EMBED_N*COMPRESS), dtype=torch.float16, device=DEVICE),
  #          #'block_b': torch.randn((BATCH_SIZE*2, 3, EMBED_N*COMPRESS, EMBED_N), dtype=torch.float16, device=DEVICE),
  #          #'block_bias_b': torch.randn((BATCH_SIZE*2, 3, EMBED_N), dtype=torch.float16, device=DEVICE),
  #          'output': torch.zeros((BATCH_SIZE*2, EMBED_N, BOARD_SIZE), dtype=torch.float16, device=DEVICE),
  #          'amplitude': torch.zeros((BATCH_SIZE*2, MUTATION_PARAMS_SIZE), dtype=torch.float16, device=DEVICE),
  #          'mutuation': torch.zeros((BATCH_SIZE*2, MUTATION_PARAMS_SIZE), dtype=torch.float16, device=DEVICE)}
  params = {'input': torch.zeros((BATCH_SIZE*2, BOARD_SIZE*4, EMBED_N), dtype=torch.float16, device=DEVICE),
            'bias': torch.zeros((BATCH_SIZE*2, EMBED_N), dtype=torch.float16, device=DEVICE),
            'output': torch.zeros((BATCH_SIZE*2, EMBED_N, BOARD_SIZE), dtype=torch.float16, device=DEVICE),
            'amplitude': torch.zeros((BATCH_SIZE*2, MUTATION_PARAMS_SIZE), dtype=torch.float16, device=DEVICE),
            'mutation': torch.zeros((BATCH_SIZE*2, MUTATION_PARAMS_SIZE), dtype=torch.float16, device=DEVICE)}
  for key in list(params.keys()):
    params[key] = params[key].uniform_(-alpha,alpha)
  #  params[key + '_mutation'] = -1 * torch.ones_like(params[key])
  players = Players(params, credits)

  perfect_params = pickle.load(open('perfect_dna.pkl', 'rb'))
  perfect_players = Players.from_params(perfect_params, bs=BATCH_SIZE*2, device=DEVICE)

  import time
  import tqdm
  pbar = tqdm.tqdm(range(200000))
  for step in pbar:
    t0 = time.time()
    games = Games(bs=BATCH_SIZE)
    x_players, o_players = swizzle_players(players, bs=BATCH_SIZE)
    t1 = time.time()
    play_games(games, x_players, o_players, err=err)
    t2 = time.time()
    if step % 100 == 0:
      val_games = Games(bs=BATCH_SIZE)
      play_games(val_games, x_players, o_players, test=True)
      writer.add_scalar('total_moves_val', val_games.total_moves, step)
    players = Players(concat_params(x_players.params, o_players.params), torch.cat([x_players.credits, o_players.credits]))
    players.mate(init_credits=init_credits, alpha=alpha)
    t3 = time.time()
    if step % 100 == 0:
      print(f'Average total moves: {games.total_moves:.2f}, avg credits of X: {x_players.credits.half().mean():.2f}, avg credits of O: {o_players.credits.half().mean():.2f}')
      writer.add_scalar('total_moves', games.total_moves, step)
      writer.add_scalar('avg_log_mutuation', players.avg_log_mutuation(), step)
      writer.add_scalar('draw_rate',(games.winners == PLAYERS.NONE).half().mean(), step)

      #writer.add_scalar('avg_log_amplitude', players.avg_log_amplitude(), step)
      string = f'swizzling took {1000*(t1-t0):.2f}ms, playing took {1000*(t2-t1):.2f}ms, mating took {1000*(t3-t2):.2f}ms'
      pbar.set_description(string)
    if step % 1000 == 0:
      pickle.dump(players.params, open('organic_dna.pkl', 'wb'))
      games = Games(bs=BATCH_SIZE*2, device=DEVICE)
      play_games(games, perfect_players, players, test=True) 
      
      print(f'Vs perfect player avg total moves: {games.total_moves:.2f}, X win rate: {(games.winners == PLAYERS.X).half().mean():.2f}, O win rate: {(games.winners == PLAYERS.O).half().mean():.2f}')
      writer.add_scalar('perfect_total_moves', games.total_moves, step)
      writer.add_scalar('perfect_loss_rate',(games.winners == PLAYERS.X).half().mean(), step)
      writer.add_scalar('perfect_draw_rate',(games.winners == PLAYERS.NONE).half().mean(), step)
    #players.credits -= players.credits.min()



  writer.close()
  
if __name__ == '__main__':
  init_credits_l = [1,2,4,8,16]
  for i in range(1100,11000):
    init_credits = 1 #init_credits_l[i -10]
    size_factor = 8
    alpha = 0.25
    err = 0
    name = f'run2_{i}'
    train_run(name=name, init_credits=init_credits, embed_n=size_factor*16, bs=10000*8//size_factor, alpha=alpha, err=err)
