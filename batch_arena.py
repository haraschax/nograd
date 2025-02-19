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
EMBED_N = 128
NOISE_SIZE = 4
MUTATION_PARAMS_SIZE = 50
INPUT_DIM = (BOARD_SIZE*3 + NOISE_SIZE) * EMBED_N
OUTPUT_DIM = EMBED_N * BOARD_SIZE
STRAIGHT_DIM = (BOARD_SIZE*3 + NOISE_SIZE) * BOARD_SIZE
BIAS_DIM = EMBED_N
GENE_MUTATION_SIZE = 10
GENE_SIZE = 128
GENE_N = 16

DNA_SIZE = GENE_N * GENE_SIZE

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
    self.illegal_movers = torch.zeros((self.bs,), dtype=torch.int8, device=self.device)
    self.update_game_over()

  def update(self, moves, player, test=False, player_dict=None):
    assert len(moves) == self.bs
    assert len(moves) == self.boards.shape[0]
    move_idxs = torch.argmax(moves, dim=1, keepdim=True)
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
    self.bs = params['dna'].shape[0]
    self.device = params['dna'].device
    self.perfect = torch.zeros(params['dna'].shape[0], device=self.device, dtype=torch.bool)
    self.params = params
    self.mutation_rate = torch.nan*torch.zeros((self.bs,), device=self.device)


  def play(self, boards, test=False, current_player=PLAYERS.X):
    boards_onehot_raw = F.one_hot(boards.long(), num_classes=3)
    boards_onehot = boards_onehot_raw.clone()
    noise = 0.5 * torch.ones((boards.shape[0], NOISE_SIZE), device=boards.device)
    noise[~self.perfect,:] = noise[~self.perfect,:].uniform_(0, 1)
    noise[:,-1] = (current_player*torch.ones_like(noise[:,-1]) - 1.5)

    inputs = torch.cat([boards_onehot.reshape((-1, BOARD_SIZE*3)), noise], dim=1)


    STATE_SIZE = 128
    CORE_SIZE = 32
    state = torch.zeros((self.bs, STATE_SIZE), device=boards.device)
    moves = torch.zeros((self.bs, BOARD_SIZE), device=boards.device)
    state[:,:BOARD_SIZE*3 + NOISE_SIZE] = inputs

    dna_by_gene = self.params['dna'].reshape(self.bs, GENE_N, GENE_SIZE)

    for i in range(GENE_N):
      bias = dna_by_gene[:, i, CORE_SIZE:2*CORE_SIZE]
      a_mask = (dna_by_gene[:, i, 2*CORE_SIZE:3*CORE_SIZE] > 0).float()
      #b_mask = (dna_by_gene[:, i, 3*CORE_SIZE:4*CORE_SIZE] > 0).float()
      #bias_mask = (dna_by_gene[:, i, 4*CORE_SIZE:5*CORE_SIZE] > 0).float()

      idx_in_a = ((dna_by_gene[:, i, 0]/2 + 0.5) * (STATE_SIZE - CORE_SIZE + 1)).to(dtype=torch.long)
      #idx_in_b = ((dna_by_gene[:, i, 1]/2 + 0.5) * (STATE_SIZE - CORE_SIZE + 1)).to(dtype=torch.long)
      out_idx = ((dna_by_gene[:, i, 3]/2 + 0.5) * (STATE_SIZE - 31)).to(dtype=torch.long) + 31
      move_idx = ((dna_by_gene[:, i, 3]/2 + 0.5) * BOARD_SIZE).to(dtype=torch.long)

      #print(move_idx.min())

      bool_relu_a = (dna_by_gene[:, i, 4] > 0).reshape((-1,1)).float()
      #bool_relu_b = (dna_by_gene[:, i, 5] > 0).reshape((-1,1)).float()
      #bool_mult = (dna_by_gene[:, i, 6] > 0).reshape((-1,1)).float()
      bool_out = (dna_by_gene[:, i, 7] > 0).reshape((-1,1)).float()

      B = state.size(0)
      device = state.device
      rel_idx = torch.arange(CORE_SIZE, device=device).unsqueeze(0)  # Shape: [1, CORE_SIZE]
      rel_idx_out = torch.arange(1, device=device).unsqueeze(0)  # Shape: [1, CORE_SIZE]
      out_cols = out_idx.unsqueeze(1) + rel_idx_out  # Shape: [B, CORE_SIZE]
      move_cols = move_idx.unsqueeze(1) + rel_idx_out  # Shape: [B, CORE_SIZE]
      a_cols = idx_in_a.unsqueeze(1) + rel_idx     # Shape: [B, CORE_SIZE]
      #b_cols = idx_in_b.unsqueeze(1) + rel_idx     # Shape: [B, CORE_SIZE]
      batch_idx = torch.arange(B, device=device).unsqueeze(1)  # Shape: [B, 1]
      #term1 = bias * bias_mask               # Shape: [B]
      #term2 = state[batch_idx, a_cols] * a_mask
      #term3 = state[batch_idx, b_cols] * b_mask
      #term2 = torch.relu(term2) * bool_relu_a + term2 * (1 - bool_relu_a)
      #term3 = torch.relu(term3) * bool_relu_b + term3 * (1 - bool_relu_b)

      #term4 = (state[batch_idx, a_cols] * state[batch_idx, b_cols] *
      #        bool_mult)
      #update = term1 + term2 + term3 + term4
      update = bias * a_mask * state[batch_idx, a_cols]
      update = bool_relu_a * torch.relu(update) + (1 - bool_relu_a) * update
      state[batch_idx, out_cols] += (1 - bool_out) * torch.sign(update.sum(dim=1).unsqueeze(1))
      moves[batch_idx, move_cols] += bool_out * update.sum(dim=1).unsqueeze(1)


    moves = torch.clone(state[:,-BOARD_SIZE:])


    moves = torch.nn.functional.normalize(moves, dim=1)
   

    #moves = torch.softmax(10*moves, dim=1)
    #if not test:
     # #moves[boards == PLAYERS.NONE] += 0.01
    #  moves /= moves.sum(dim=1)[:, None]
    #  sample_idx = torch.multinomial(moves, num_samples=1)
    #  moves = F.one_hot(sample_idx.long(), num_classes=9)[:,0,:].float()
    #  return moves
    #else:
    #  return moves
    #if not test:
    #  moves[boards == PLAYERS.NONE] += 1e8 * torch.ones_like(moves[boards == PLAYERS.NONE]) * (torch.rand_like(moves[boards == PLAYERS.NONE]) < 0.003).float()
    return moves

  def trans_mut(self):
    return torch.sigmoid(self.params['trans_mutation'].sum(dim=1)).mean().float().item()

  def avg_log_mutuation_size(self):
    return torch.sigmoid((self.params['dna_mutation'].sum(dim=1))).mean().float().item()

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


    
    new_params = {k:v.clone() for k,v in self.params.items()}
    indices = torch.randperm(len(can_mate))
    trans_mut_logit = self.params['trans_mutation'].sum(dim=1)[:,None].expand((-1, GENE_N)).clone()
    #trans_mut_logit += self.params['dna'].reshape((self.bs, GENE_N, GENE_SIZE))[:,:,-2*GENE_MUTATION_SIZE:-GENE_MUTATION_SIZE].sum(dim=2).clone()

    trans_mut_rates = torch.torch.sigmoid(trans_mut_logit)/4
    mix_mutation = (torch.rand_like(self.params['dna'].reshape((self.bs, GENE_N, GENE_SIZE))[:,:,0]) < trans_mut_rates)[:,:,None].float()
    pre_mixed_params = self.params['dna'].reshape((self.bs, GENE_N, GENE_SIZE))
    new_params['dna'][can_mate] = (pre_mixed_params[can_mate]  * (1 - mix_mutation[can_mate]) + pre_mixed_params[can_mate][indices] * mix_mutation[can_mate]).reshape((-1, GENE_N*GENE_SIZE))
    '''
    '''

    for key in self.params:
      if key in ['input', 'output', 'bias']:
        continue
      if 'mutation' in key:
        mutation_logit = self.params['mutation_mutation'].sum(dim=1)
        mutation_rate = torch.sigmoid(mutation_logit)[can_mate,None]
        param = torch.clone(self.params[key])[can_mate]
        mutation = (torch.rand_like(param) < mutation_rate).float()
        param = (1 - mutation) * param + mutation * torch.zeros_like(param).uniform_(-1, 1)
        self.params[key][dead] = param[:len(dead)]
      else:
        mutation_logit = self.params['dna_mutation'].sum(dim=1)[:,None].expand((-1, GENE_N)).clone()
        #mutation_logit += self.params['dna'].reshape((self.bs, GENE_N, GENE_SIZE))[:,:,-GENE_MUTATION_SIZE:].sum(dim=2).clone()
        mutation_rate = torch.sigmoid(mutation_logit)[can_mate][:,:,None]
        param = torch.clone(new_params[key])[can_mate].reshape((-1, GENE_N, GENE_SIZE))
        mutation = (torch.rand_like(param) < mutation_rate).float()
        param = (1 - mutation) * param + mutation * torch.zeros_like(param).uniform_(-1, 1)
        self.params[key][dead] = param[:len(dead)].reshape((-1, GENE_N*GENE_SIZE))


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
      #other_player = next_player(player)
      #player_dict[player].credits[(games.winners == player) & (games.illegal_movers != other_player)] += 1.0
      player_dict[player].credits[(games.winners == player)] += 1.0
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

  BATCH_SIZE = bs
  params = {}
  params['dna'] = torch.zeros((BATCH_SIZE*2, GENE_SIZE*GENE_N), dtype=torch.float, device=DEVICE).uniform_(-1, 1)
  params['dna_mutation'] = torch.zeros((BATCH_SIZE*2, MUTATION_PARAMS_SIZE), dtype=torch.float, device=DEVICE).uniform_(-1, 1)
  params['trans_mutation'] = torch.zeros((BATCH_SIZE*2, MUTATION_PARAMS_SIZE), dtype=torch.float, device=DEVICE).uniform_(-1, 1)
  params['mutation_mutation'] = torch.zeros((BATCH_SIZE*2, MUTATION_PARAMS_SIZE), dtype=torch.float, device=DEVICE).uniform_(-1, 1)
    
  perfect_params = pickle.load(open('perfect_dna.pkl', 'rb'))
  perfect_params['dna'] = torch.zeros((BATCH_SIZE, GENE_SIZE*GENE_N), dtype=torch.float, device=DEVICE)
  perfect_players = Players.from_params(perfect_params, bs=BATCH_SIZE, device=DEVICE)
  perfect_players.perfect = torch.ones(BATCH_SIZE, device=DEVICE, dtype=torch.bool)

  players = Players(params)
  players.credits = torch.ones((BATCH_SIZE*2,), device=DEVICE) * INIT_CREDS


  import time
  import tqdm
  pbar = tqdm.tqdm(range(500000))
  a_players, b_players = swizzle_players(players, bs=BATCH_SIZE)

  for step in pbar:
    
    t1 = time.time()
    a_wins = torch.zeros((BATCH_SIZE,), dtype=torch.int8, device=DEVICE)
    b_wins = torch.zeros((BATCH_SIZE,), dtype=torch.int8, device=DEVICE)

    games = Games(bs=BATCH_SIZE)
    play_games(games, a_players, b_players)
    a_wins = (games.winners == PLAYERS.X)
    b_wins = (games.winners == PLAYERS.O)

    mut_rate = a_players.avg_log_mutuation_size()
    trans_mut = a_players.trans_mut()
    t2 = time.time()
    concat_players = Players(concat_params(a_players.params, b_players.params))
    concat_players.credits = torch.cat([a_players.credits, b_players.credits])
    if step % 100 == 0:
      print(f'mean a_player credits: {a_players.credits.mean():2f} and mean b_player credits: {b_players.credits.mean():.2f}')

    #concat_players.credits += INIT_CREDS - concat_players.credits.mean()
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

      print(f'Average total moves: {games.total_moves:.2f}, Average mutuation rate: {mut_rate:.1e}')
      writer.add_scalar('total_moves', games.total_moves, step)
      writer.add_scalar('avg_log_mutuation', mut_rate, step)
      writer.add_scalar('trans_mutuation', trans_mut, step)
      writer.add_scalar('draw_rate',(a_wins == b_wins).float().mean(), step)

    if step % 100 == 0:
      string = f'swizzling took {1000*(t4-t3):.2f}ms, playing took {1000*(t2-t1):.2f}ms, mating took {1000*(t3-t2):.2f}ms'
      pbar.set_description(string)
    if step % 1000 == 0:
      print('Saving...')
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
  for i in range(40,2000):
    bs = 100000
    name = f'run_{i}'
    train_run(name=name, embed_n=EMBED_N, bs=bs)
