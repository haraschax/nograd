#!/usr/bin/env python
import pickle
import torch
import math
import numpy as np
torch.set_grad_enabled(False)
import torch.nn.functional as F
from helpers import PLAYERS, next_player, BOARD_SIZE
from tensorboardX import SummaryWriter
from torch.nn import functional as F
#import torch_dct as dct

MAX_MOVES = 10
BATCH_SIZE = 10
INIT_CREDS = 1
EMBED_N = 512
NOISE_SIZE = 16
MUTATION_PARAMS_SIZE = 50
INPUT_DIM = (BOARD_SIZE*3 + NOISE_SIZE*2) * EMBED_N
OUTPUT_DIM = EMBED_N * BOARD_SIZE
BIAS_DIM = EMBED_N
RUN_DIM = INPUT_DIM + OUTPUT_DIM + BIAS_DIM

GENE_N = 512
META_CORE = 64
META_INPUT = META_CORE
META_OUTPUT = META_CORE
META_A = META_INPUT*META_CORE
META_B = META_CORE*META_CORE*3
META_BIAS = META_CORE
META_C = META_CORE*META_OUTPUT
META_BIAS2 = META_CORE*3
META_SIZE = META_A + META_B + META_BIAS + META_C + META_BIAS2

DNA_SIZE = GENE_N * META_CORE

print(DNA_SIZE, META_SIZE)
print(GENE_N * META_OUTPUT, RUN_DIM)

#assert DNA_SIZE > META_SIZE
#assert GENE_N * META_OUTPUT > RUN_DIM + MUTATION_PARAMS_SIZE

DEVICE = 'cuda'

def sinusoidal_positional_encoding(seq_length, embedding_dim):
    position = torch.arange(seq_length, dtype=torch.float).unsqueeze(1)

    div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))

    pos_enc = torch.zeros(seq_length, embedding_dim)
    pos_enc[:, 0::2] = torch.sin(position * div_term)  # Even indices
    pos_enc[:, 1::2] = torch.cos(position * div_term)  # Odd indices

    return pos_enc

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
    self.illegal_movers = torch.zeros((self.bs,), dtype=torch.int8, device=self.device)
    self.update_game_over()

  def update(self, moves, player, test=False, player_dict=None):
    assert len(moves) == self.bs
    assert len(moves) == self.boards.shape[0]
    move_idxs = torch.argmax(moves, dim=1, keepdim=True)

    #if not test:
    #  random_moves = torch.rand(size=move_idxs.shape, device=moves.device) < 0.01
    #  move_idxs[random_moves] = torch.randint(0, 9, (self.bs, 1), device=self.device)[random_moves]
    #print(self.illegal_movers[:10], self.winners[:10])
    assert (self.illegal_movers[self.game_over == 0] == PLAYERS.NONE).all()

    illegal_moves = (self.boards.gather(1, move_idxs) != PLAYERS.NONE).reshape(-1)
    self.illegal_movers[(self.game_over == 0) & illegal_moves] = PLAYERS.O if player == PLAYERS.O else PLAYERS.X
    self.winners[illegal_moves & (self.game_over == 0)] = PLAYERS.O if player == PLAYERS.X else PLAYERS.X
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

  @property
  def _total_moves(self):
    return (self.boards != PLAYERS.NONE).sum(dim=1).float()

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
    self.perfect = torch.zeros(params['input'].shape[0], device=self.device, dtype=torch.bool)
    self.params = params
    self.reset_state()
    self.mutation_rate = torch.nan*torch.zeros((self.bs,), device=self.device)

  def reset_state(self):
    self.state = 0.5 * torch.ones((self.bs, NOISE_SIZE), device=self.device)

  def embryogenesis(self):
    new_params = {}
    matrix_A = self.params['dna2'][:, :META_A].reshape((self.bs, META_INPUT, META_CORE))
    matrix_B = self.params['dna2'][:, META_A:META_A+META_B].reshape((self.bs, META_CORE, META_CORE*3))
    bias = self.params['dna2'][:, META_A+META_B:META_A+META_B+META_BIAS]

    matrix_C = self.params['dna2'][:, META_A+META_B+META_BIAS:META_A+META_B+META_BIAS+META_C].reshape((self.bs, META_CORE, META_OUTPUT))
    bias2 = self.params['dna2'][:, META_A+META_B+META_BIAS+META_C:META_A+META_B+META_BIAS+META_C+META_BIAS2]

    dna_by_genes =  self.params['dna'].reshape(self.bs, -1, META_INPUT)
    #dna_w_noise = torch.cat([dna_by_genes, torch.zeros_like(dna_by_genes).uniform_(-1,1)], dim=2)
    #print(dna_w_noise.shape)
    #pos_enc = sinusoidal_positional_encoding(dna_by_genes.shape[1], dna_by_genes.shape[2])
    #print(pos_enc)
    #dna_by_genes = dna_by_genes + pos_enc[None,:,:].to(device=dna_by_genes.device)
    embed = torch.einsum('bji, bnj->bni', matrix_A, dna_by_genes)
    embed = embed + bias[:,None,:]
    embed = torch.relu(embed)

    embed = torch.nn.functional.normalize(embed, dim=2)

    qkv = torch.einsum('bji, bnj->bni', matrix_B, embed)
    qkv = qkv + bias2[:,None,:]
    q, k, v = qkv.chunk(3, dim=2)

    embed = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    embed = torch.einsum('bji, bnj->bni', matrix_C, embed)
    #print(embed.abs().mean())

    unrolled_flat = embed.reshape((self.bs, -1)) + self.params['dna']
    #unrolled_flat = self.params['dna']
    #unrolled_flat = torch.nn.functional.normalize(unrolled_flat)
    #print(unrolled_flat.shape, DNA_SIZE, embed.shape)
 
    #print(unrolled_flat.shape, self.params['dna'].shape, GENE_N*META_INPUT//2)
    #print(unrolled_flat[:, :GENE_N*META_INPUT//2].abs().mean())
    #mutation = (torch.rand_like(self.params['dna']) < 1e-3).float()
    #new_params['dna'] = (1 - mutation) * self.params['dna'] + mutation * torch.zeros_like(self.params['dna']).uniform_(-1, 1)
    mutation_logit = self.params['dna2'][:, -MUTATION_PARAMS_SIZE:].sum(dim=1)
    mutation_rate = torch.sigmoid(mutation_logit)[:,None]
    #print(mutation_rate.mean().float().item())
    #print(mutation_logit_by_gene[:10,:10])
    mutation = (torch.rand_like(self.params['dna']) < mutation_rate).float()
    new_params['dna'] = (1 - mutation) * self.params['dna'] + mutation * torch.zeros_like(self.params['dna']).uniform_(-1, 1)
    mutation = (torch.rand_like(self.params['dna']) < mutation_rate).float()
    #new_params['dna2'] = (1 - mutation) * self.params['dna2'] + mutation * torch.zeros_like(self.params['dna2']).uniform_(-1, 1)

    #torch.nn.functional.normalize(self.params['dna'] + unrolled_flat[:, :GENE_N*META_INPUT//2])

    new_params['input'] = unrolled_flat[:, :INPUT_DIM]
    new_params['output'] = unrolled_flat[:, INPUT_DIM:INPUT_DIM+OUTPUT_DIM]
    new_params['bias'] = unrolled_flat[:, INPUT_DIM+OUTPUT_DIM:INPUT_DIM+OUTPUT_DIM+BIAS_DIM]
    return new_params


  def play(self, boards, test=False, current_player=PLAYERS.X):
    boards_onehot = F.one_hot(boards.long(), num_classes=3).reshape((boards.shape[0], -1))
    noise = 0.5 * torch.ones((boards.shape[0], NOISE_SIZE), device=boards.device)
    noise[~self.perfect,:8] = noise[~self.perfect,:8] * current_player * 2 - 1.5 #noise.uniform_(0, 1)[~self.perfect]
    noise[~self.perfect,8:] = noise[~self.perfect,8:].uniform_(0, 1)

    inputs = torch.cat([boards_onehot.reshape((-1, BOARD_SIZE*3)), self.state, noise], dim=1)
    matrix_A = self.params['input'].reshape((self.bs, BOARD_SIZE*3 + NOISE_SIZE*2, EMBED_N))
    matrix_B = self.params['output'].reshape((self.bs,  EMBED_N, BOARD_SIZE))

    embed = torch.einsum('bji, bj->bi', matrix_A, inputs)
    embed = embed + self.params['bias']
    embed = torch.relu(embed)
    out = torch.einsum('bji, bj->bi', matrix_B, embed)

    moves =  out[:,:BOARD_SIZE]
    moves = torch.nn.functional.normalize(moves, dim=1)
    #moves[boards != PLAYERS.NONE] = -1e9
    if not test:
      #moves += (torch.rand_like(moves[:,0]) < 0.5)[:,None].float() * torch.rand_like(moves) * .1
      moves[boards == PLAYERS.NONE] += 1e8 * torch.ones_like(moves[boards == PLAYERS.NONE]) * (torch.rand_like(moves[boards == PLAYERS.NONE]) < 0.05).float()
      #print( 1e8 * torch.ones_like(moves[boards == PLAYERS.NONE]) * (torch.rand_like(moves[boards == PLAYERS.NONE]) < 0.5).float())
      #print((moves.reshape((-1,)) > 1e4).sum() / moves.reshape((-1,)).shape[0])
    #self.state = out[:,BOARD_SIZE:]
    return moves

  def trans_mut(self):
    return torch.sigmoid(self.params['trans_mutation'].sum(dim=1)).mean().float().item()

  def avg_log_mutuation_size(self):
    return torch.sigmoid((self.params['mutation_mutation'].sum(dim=1))).mean().float().item()

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
    
    '''
    indices = torch.randperm(len(can_mate))
    trans_mut_logit = self.params['trans_mutation'].sum(dim=1)[can_mate]
    trans_mut_rates = torch.torch.sigmoid(trans_mut_logit)/10
    #trans_mut_rates[:] = 0.0
    mixed_params = {}
    for key in self.params:
      param = torch.clone(self.params[key])
      pre_mixed_params = param[can_mate]
      if 'mutation' not in key and False:
        mix_mutation = (torch.rand_like(pre_mixed_params) > trans_mut_rates[:,None]).float()
        mixed_params[key] = pre_mixed_params  * mix_mutation + pre_mixed_params[indices] * (1 - mix_mutation)
      else:
        mixed_params[key] = pre_mixed_params

    #new_params = self.embryogenesis()
    for key in self.params:
      if 'mutation' in key:
        mutation_logit = self.params['mutation_mutation'].sum(dim=1)
      else:
        mutation_logit = self.params[f'{key}_mutation'].sum(dim=1)
      mutation_rate = torch.sigmoid(mutation_logit)[can_mate,None]
      param = torch.clone(mixed_params[key])
      mutation = (torch.rand_like(param) < mutation_rate).float()
      param = (1 - mutation) * param + mutation * torch.zeros_like(param).uniform_(-1, 1)
      #param = param + mutation_rate * torch.zeros_like(param).uniform_(-1, 1)
      self.params[key][dead] = param[:len(dead)]
    '''
    new_params = self.embryogenesis()
    for key in self.params:
      if 'dna' not in key:
        self.params[key][dead] = new_params[key][can_mate][:len(dead)]


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
  x_players = Players(splice_params(players.params, indices[:bs]))
  x_players.credits = players.credits[indices[:bs]]
  o_players = Players(splice_params(players.params, indices[bs:]))
  o_players.credits = players.credits[indices[bs:]]
  return x_players, o_players

def train_run(name='', embed_n=EMBED_N, bs=BATCH_SIZE):
  writer = SummaryWriter(f'runs/{name}')

  EMBED_N = embed_n
  BATCH_SIZE = bs

  params = {'input': torch.zeros((BATCH_SIZE*2, INPUT_DIM), dtype=torch.float, device=DEVICE),
            'bias': torch.zeros((BATCH_SIZE*2, BIAS_DIM), dtype=torch.float, device=DEVICE),
            'output': torch.zeros((BATCH_SIZE*2, OUTPUT_DIM), dtype=torch.float, device=DEVICE)}
  params['mutation'] = torch.zeros((BATCH_SIZE*2, MUTATION_PARAMS_SIZE), dtype=torch.float, device=DEVICE).uniform_(-1, 1)
  for key in list(params.keys()):
    params[f'{key}_mutation'] = torch.zeros((BATCH_SIZE*2, MUTATION_PARAMS_SIZE), dtype=torch.float, device=DEVICE).uniform_(-1, 1)
  params['trans_mutation'] = torch.zeros((BATCH_SIZE*2, MUTATION_PARAMS_SIZE), dtype=torch.float, device=DEVICE).uniform_(-1, 1)

  #params['dna2'] = torch.zeros((BATCH_SIZE*2, DNA_SIZE), dtype=torch.float, device=DEVICE).uniform_(-1, 1)
  #for i in range(9):
  #  params[f'input_{i}'] = torch.zeros((BATCH_SIZE*2, (BOARD_SIZE*3 + NOISE_SIZE*2) * EMBED_N), dtype=torch.float, device=DEVICE)
  #  params[f'bias_{i}'] = torch.zeros((BATCH_SIZE*2, EMBED_N), dtype=torch.float, device=DEVICE)
  #  params[f'output_{i}'] = torch.zeros((BATCH_SIZE*2, EMBED_N *(BOARD_SIZE)), dtype=torch.float, device=DEVICE)
    #params[f'inter_block_{i}'] = torch.zeros((BATCH_SIZE*2, EMBED_N, EMBED_N), dtype=torch.float, device=DEVICE)
    #params[f'inter_bias_{i}'] = torch.zeros((BATCH_SIZE*2, EMBED_N), dtype=torch.float, device=DEVICE)
  #for key in list(params.keys()):
  # params[key + '_mutation'] = torch.zeros(list(params[key].shape) + [10,], dtype=torch.float, device=DEVICE).uniform_(-1, 1)

  #  meta_shape = tuple(list(params[key].shape) + [META_INPUT])
  #  params[key + '_meta'] = torch.zeros(meta_shape, dtype=torch.float, device=DEVICE).uniform_(-1, 1)
  #params['embryogenesis'] = torch.zeros((BATCH_SIZE*2, META_SIZE), dtype=torch.float, device=DEVICE).uniform_(-1,1)
  #params['embryogenesis_mutation'] = torch.zeros((BATCH_SIZE*2, MUTATION_PARAMS_SIZE), dtype=torch.float, device=DEVICE).uniform_(-1,1)
  #params['trans_mutation'] = torch.zeros((BATCH_SIZE*2, MUTATION_PARAMS_SIZE), dtype=torch.float, device=DEVICE).uniform_(-1,1)
  #params['mutation_mutation'] = torch.zeros((BATCH_SIZE*2, MUTATION_PARAMS_SIZE), dtype=torch.float, device=DEVICE).uniform_(-1,1)
    
  perfect_params = pickle.load(open('perfect_dna.pkl', 'rb'))
  max_param = max([abs(perfect_params[key]).max() for key in perfect_params])
  perfect_players = Players.from_params(perfect_params, bs=BATCH_SIZE, device=DEVICE)
  perfect_players.credits = torch.ones(BATCH_SIZE, device=DEVICE, dtype=torch.bool)

  PERFECT_AMOUNT = BATCH_SIZE//10
  for key in perfect_params:
    params[key][:PERFECT_AMOUNT] = torch.cat([perfect_params[key] for _ in range(PERFECT_AMOUNT)], dim=0)/max_param
    params[key][:PERFECT_AMOUNT] += torch.randn_like(params[key][:PERFECT_AMOUNT]) * 1e-1
  players = Players(params)
  players.credits = torch.ones((BATCH_SIZE*2,), device=DEVICE) * INIT_CREDS


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
    #mut_rate = a_players.avg_log_mutuation()
    mut_size = a_players.avg_log_mutuation_size()
    trans_mut = a_players.trans_mut()
    t2 = time.time()
    concat_players = Players(concat_params(a_players.params, b_players.params))
    concat_players.credits = torch.cat([a_players.credits, b_players.credits])
    t3 = time.time()
    concat_players.mate()
    a_players, b_players = swizzle_players(concat_players, bs=BATCH_SIZE)
    t4 = time.time()
    if step % 100 == 0:
      writer.add_scalar('total_moves_val', games.total_moves, step)
      assert ((games.illegal_movers != games.winners) | (games._total_moves == 9)).all()
      writer.add_scalar('o_illegal_move_rate', (games.illegal_movers == PLAYERS.O).sum()/BATCH_SIZE, step)
      writer.add_scalar('x_illegal_move_rate', (games.illegal_movers == PLAYERS.X).sum()/BATCH_SIZE, step)
      writer.add_scalar('o_win_rate', ((games.winners == PLAYERS.O) & (games.illegal_movers != PLAYERS.X)).sum()/BATCH_SIZE, step)
      writer.add_scalar('x_win_rate', ((games.winners == PLAYERS.X) & (games.illegal_movers != PLAYERS.O)).sum()/BATCH_SIZE, step)

      print(f'Average total moves: {games.total_moves:.2f}, Average mutuation rate: {mut_size:.1e}')
      writer.add_scalar('total_moves', games.total_moves, step)
      #writer.add_scalar('avg_log_mutuation', mut_rate, step)
      writer.add_scalar('avg_log_mutuation_size', mut_size, step)
      writer.add_scalar('trans_mutuation', trans_mut, step)
      writer.add_scalar('draw_rate',(a_wins == b_wins).float().mean(), step)

    if step % 100 == 0:
      string = f'swizzling took {1000*(t4-t3):.2f}ms, playing took {1000*(t2-t1):.2f}ms, mating took {1000*(t3-t2):.2f}ms'
      pbar.set_description(string)
    if step % 1000 == 0:
      print('Saving...')
      print(a_players.params['input'].shape)
      pickle.dump(a_players.params, open('organic_dna.pkl', 'wb'))

      a_games = Games(bs=BATCH_SIZE, device=DEVICE)
      play_games(a_games, a_players, perfect_players, test=True)
      b_games = Games(bs=BATCH_SIZE, device=DEVICE)
      play_games(b_games, perfect_players, b_players, test=True)
      
      perfect_total_moves = (a_games.total_moves + b_games.total_moves)/2
      print(f'Vs perfect player avg total moves: {perfect_total_moves:.2f}')
      writer.add_scalar('avg_moves_vs_perfect_player', perfect_total_moves, step)

  writer.close()
  
if __name__ == '__main__':
  for i in range(120,2000):
    bs = 5000
    name = f'run_{i}'
    train_run(name=name, embed_n=EMBED_N, bs=bs)
