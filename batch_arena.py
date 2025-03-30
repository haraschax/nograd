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
CORE_SIZE = 8
GENE_I = 16
GENE_J = 16
GENE_N = GENE_I * GENE_J
STATE_SIZE = 128
BOOLS_SIZE = 8
GENE_SIZE = BOOLS_SIZE + 4*CORE_SIZE + GENE_MUTATION_SIZE

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
    self.mutation = torch.zeros((self.bs,), device=self.device)
    self.trans_mutation = torch.zeros((self.bs,), device=self.device)/4


  def run_dna(self, dna_by_gene, input_vector):
    for i in range(dna_by_gene.shape[1]):
      state_expanded = torch.zeros((self.bs * GENE_I, STATE_SIZE), device=input_vector.device)

      bias = dna_by_gene[:, i, BOOLS_SIZE:BOOLS_SIZE+CORE_SIZE]
      a_mask = (dna_by_gene[:, i, BOOLS_SIZE+CORE_SIZE:BOOLS_SIZE+2*CORE_SIZE] > 0).float()
      b_mask = (dna_by_gene[:, i, BOOLS_SIZE+2*CORE_SIZE:BOOLS_SIZE+3*CORE_SIZE] > 0).float()
      bias_mask = (dna_by_gene[:, i, BOOLS_SIZE+3*CORE_SIZE:BOOLS_SIZE+4*CORE_SIZE] > 0).float()

      idx_in_a = ((dna_by_gene[:, i, 0]/2 + 0.5) * (STATE_SIZE - CORE_SIZE + 1)).to(dtype=torch.long)
      idx_in_b = ((dna_by_gene[:, i, 1]/2 + 0.5) * (STATE_SIZE - CORE_SIZE + 1)).to(dtype=torch.long)
      out_idx = ((dna_by_gene[:, i, 2]/2 + 0.5) * (STATE_SIZE - 31)).to(dtype=torch.long) + 31
      bool_relu_a = (dna_by_gene[:, i, 3] > 0).reshape((-1,1)).float()
      out_sign = (dna_by_gene[:, i, 4] > 0).reshape((-1,1)).float()*2 - 1
      bool_sum = (dna_by_gene[:, i, 5] > 0).reshape((-1,1)).float()
      bool_mult = (dna_by_gene[:, i, 6] > 0).reshape((-1,1)).float()
      bool_out = (dna_by_gene[:, i, 7] > 0).reshape((-1,1)).float()

      B = self.bs * GENE_I
      device = input_vector.device
      rel_idx = torch.arange(CORE_SIZE, device=device).unsqueeze(0)  # Shape: [1, CORE_SIZE]
      rel_idx_out = torch.arange(1, device=device).unsqueeze(0)  # Shape: [1, CORE_SIZE]
      out_cols = out_idx.unsqueeze(1) + rel_idx_out  # Shape: [B, CORE_SIZE]
      #move_cols = move_idx.unsqueeze(1) + rel_idx_out  # Shape: [B, CORE_SIZE]
      a_cols = idx_in_a.unsqueeze(1) + rel_idx     # Shape: [B, CORE_SIZE]
      b_cols = idx_in_b.unsqueeze(1) + rel_idx     # Shape: [B, CORE_SIZE]
      batch_idx = torch.arange(B, device=device).unsqueeze(1)//GENE_I  # Shape: [B, 1]
      batch_exapanded_idx = torch.arange(B, device=device).unsqueeze(1)  # Shape: [B, 1]
      #term1 = bias * bias_mask               # Shape: [B]
      a = input_vector[batch_idx, a_cols] * a_mask
      b = input_vector[batch_idx, b_cols] * b_mask
      #term2 = torch.relu(term2) * bool_relu_a + term2 * (1 - bool_relu_a)
      #term3 = torch.relu(term3) * bool_relu_b + term3 * (1 - bool_relu_b)

      #term4 = (state[batch_idx, a_cols] * state[batch_idx, b_cols] *
      #        bool_mult)
      #update = term1 + term2 + term3 + term4
      #update = bias_mask * bias * a
      #update = bias_mask * bias * a
      update = bias_mask * bias * bool_sum * (a + b)
      update += bias_mask * bias * bool_mult * (a * b)
      #update += a_mask * input_vector[batch_idx, a_cols]

      #update = bias_mask * bias + a_mask * input_vector[batch_idx, a_cols]
      update = out_sign * (bool_relu_a * torch.relu(update) + (1 - bool_relu_a) * update)
      #print(update, bool_out)
      state_expanded[batch_exapanded_idx, out_cols] += (1 - bool_out) * (update.sum(dim=1).unsqueeze(1))

      #print(state_expanded)
      #print(state_expanded.reshape((self.bs, GENE_I, -1)).sum(dim=1))
      #print(input_vector)
      input_vector += torch.sign(state_expanded.reshape((self.bs, GENE_I, -1)).sum(dim=1))
    return input_vector

  def embryogenesis(self):
    mut_mut_logit = self.params['mutation_mutation'].sum(dim=1)
    #trans_mut_logit += self.params['dna'].reshape((self.bs, GENE_N, GENE_SIZE))[:,:,-2*GENE_MUTATION_SIZE:-GENE_MUTATION_SIZE].sum(dim=2).clone()

    self.mutation_mutation = torch.torch.sigmoid(mut_mut_logit)

    #state = torch.zeros((self.bs, STATE_SIZE), device=self.params['dna'].device)
    #dna_by_gene = self.params['dna'].reshape(self.bs * GENE_I, GENE_J*2, GENE_SIZE)[:,:GENE_J,:]
    #state = self.run_dna(dna_by_gene, state)
    #print(state[:,-CORE_SIZE:].sum(dim=1))
    #self.mutation = torch.sigmoid(state[:,-64:-32].sum(dim=1)/5)
    #self.mutation = torch.sigmoid(state[:,-32:].sum(dim=1)/5)
    #probs = torch.softmax(state[:, -32:], dim=1)
    #sampled_index = torch.multinomial(probs, num_samples=1).squeeze(1)
    trans_mut_logit = self.params['trans_mutation'].sum(dim=1)
    #trans_mut_logit += self.params['dna'].reshape((self.bs, GENE_N, GENE_SIZE))[:,:,-2*GENE_MUTATION_SIZE:-GENE_MUTATION_SIZE].sum(dim=2).clone()

    self.trans_mutation = torch.torch.sigmoid(trans_mut_logit)/4
    #self.trans_mutation = 1e-12*torch.exp(-torch.sum(state[:, -32:], dim=1).float())
    #self.trans_mutation[:] = 1e-3 #torch.argmax(state[:,-32:], dim=1)
    #print(self.mutation)

    #self.mutation[:] = 1e-2
    #self.trans_mutation = torch.sigmoid(state[:,-32:].sum(dim=1)/5)
    #probs = torch.softmax(state[:, -64:-32]*5, dim=1)
    #sampled_index = torch.multinomial(probs, num_samples=1).squeeze(1)
    mut_logit = self.params['mutation'].sum(dim=1)
    self.mutation = torch.torch.sigmoid(mut_logit)

    #dna_by_gene = self.params['dna'].reshape(self.bs * GENE_I, GENE_J, GENE_SIZE)
    #perm = torch.argsort(torch.rand(self.bs * GENE_I, GENE_J, device=DEVICE), dim=1)
    #perm_expanded = perm.unsqueeze(2).expand(self.bs * GENE_I, GENE_J, GENE_SIZE)
    #self.params['dna'] = dna_by_gene.gather(1, perm_expanded).reshape(self.params['dna'].shape)

  def play(self, boards, test=False, current_player=PLAYERS.X):
    boards_onehot_raw = F.one_hot(boards.long(), num_classes=3)
    boards_onehot = boards_onehot_raw.clone()
    noise = 0.5 * torch.ones((boards.shape[0], NOISE_SIZE), device=boards.device)
    noise[~self.perfect,:] = noise[~self.perfect,:].uniform_(0, 1)
    noise[:,-1] = (current_player*torch.ones_like(noise[:,-1]) - 1.5)

    inputs = torch.cat([boards_onehot.reshape((-1, BOARD_SIZE*3)), noise], dim=1)


    state = torch.zeros((self.bs, STATE_SIZE), device=boards.device)
    moves = torch.zeros((self.bs, BOARD_SIZE), device=boards.device)
    state[:,:BOARD_SIZE*3 + NOISE_SIZE] = inputs

    dna_by_gene = self.params['dna'].reshape(self.bs * GENE_I, GENE_J, GENE_SIZE)
    state = self.run_dna(dna_by_gene, state)

    moves = torch.clone(state[:,-BOARD_SIZE:])# * 20 * torch.clip(torch.sigmoid(self.params['fake_mutation'].sum(dim=1)/10)[:,None], 1/20, 1)
    probs = F.softmax(3*moves, dim=1)
    moves = probs
    #sampled_indices = torch.multinomial(probs, num_samples=1)
    #moves = F.one_hot(sampled_indices.squeeze(1), num_classes=moves.size(1)).float()
    #moves = F.softmax(moves, dim=1)
    #moves = moves * torch.rand_like(moves)
    
    if not test:
      moves[boards == PLAYERS.NONE] += 1e8 * torch.ones_like(moves[boards == PLAYERS.NONE]) * (torch.rand_like(moves[boards == PLAYERS.NONE]) < 0.003).float()
    return moves

  def avg_trans_mutation(self):
    return math.log(self.trans_mutation.mean().float().item())

  def avg_mutation(self):
    return math.log(self.mutation.mean().float().item())

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

    self.embryogenesis()


    new_params = {k:v.clone() for k,v in self.params.items()}
    indices = torch.randperm(len(can_mate))
    trans_mut_rates = self.trans_mutation[:,None].expand((-1, GENE_N)).clone()
    mix_mutation = (torch.rand_like(self.params['dna'].reshape((self.bs, GENE_N, GENE_SIZE))[:,:,0]) < trans_mut_rates)[:,:,None].float()
    pre_mixed_params = self.params['dna'].reshape((self.bs, GENE_N, GENE_SIZE))
    new_params['dna'][can_mate] = (pre_mixed_params[can_mate]  * (1 - mix_mutation[can_mate]) + pre_mixed_params[can_mate][indices] * mix_mutation[can_mate]).reshape((-1, GENE_N*GENE_SIZE))

    for key in self.params:
      if 'mutation' in key:
        #mutation_logit = self.params['mutation_mutation'].sum(dim=1)
        mutation_rate = self.mutation_mutation.clone()[can_mate,None]
        param = torch.clone(self.params[key])[can_mate]
        #param = param + torch.rand_like(param) * mutation_rate
        mutation = (torch.rand_like(param) < mutation_rate).float()
        param = (1 - mutation) * param + mutation * torch.zeros_like(param).uniform_(-1, 1)
        self.params[key][dead] = param[:len(dead)]
      else:
        #mutation_logit = self.params['dna_mutation'].sum(dim=1)[:,None].expand((-1, GENE_N)).clone()
        #mutation_mult = torch.sigmoid(self.params['dna'].reshape((-1, GENE_I*GENE_J, GENE_SIZE))[:,:,-GENE_MUTATION_SIZE:].sum(dim=2).clone())
        mutation_rate = (self.mutation[:,None].expand((-1, GENE_N)).clone())[can_mate][:,:,None]
        #mutation_rate = (self.mutation[:,None].expand((-1, GENE_N)).clone())[can_mate][:,:,None]
        param = torch.clone(new_params[key])[can_mate].reshape((-1, GENE_N, GENE_SIZE))
        #mutation = (torch.rand_like(param[:,:,:1]) < mutation_rate).float()
        mutation = (torch.rand_like(param) < mutation_rate).float()
        param = (1 - mutation) * param + mutation * torch.zeros_like(param).uniform_(-1, 1)
        self.params[key][dead] = param[:len(dead)].reshape((-1, GENE_N*GENE_SIZE))

def play_games(games, x_players, o_players, test=False):
  player_dict = {PLAYERS.X: x_players, PLAYERS.O: o_players}
  current_player = PLAYERS.X
  first = True
  while True:

    moves = player_dict[current_player].play(games.boards, test=test, current_player=current_player)
    if first:
      max_indices = torch.argmax(moves, dim=1)

      moves_one_shot = F.one_hot(max_indices, num_classes=moves.size(1)).float()
      assert torch.all(moves_one_shot.sum(dim=1) <= 1)
      games.first_move_optimal = (moves_one_shot == torch.tensor([1.0,-1.,1.0,
                                                        -1.0,-1.0,-1.0,
                                                        1.0,-1.0,1.0], device=moves_one_shot.device)).sum(dim=1).float()
      first = False
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
  params['mutation'] = torch.zeros((BATCH_SIZE*2, MUTATION_PARAMS_SIZE), dtype=torch.float, device=DEVICE).uniform_(-1, 1)
  params['trans_mutation'] = torch.zeros((BATCH_SIZE*2, MUTATION_PARAMS_SIZE), dtype=torch.float, device=DEVICE).uniform_(-1, 1)
  params['mutation_mutation'] = torch.zeros((BATCH_SIZE*2, MUTATION_PARAMS_SIZE), dtype=torch.float, device=DEVICE).uniform_(-1, 1)
  params['fake_mutation'] = torch.zeros((BATCH_SIZE*2, MUTATION_PARAMS_SIZE), dtype=torch.float, device=DEVICE).uniform_(-1, 1)

    
  perfect_params = pickle.load(open('perfect_dna.pkl', 'rb'))
  perfect_params['dna'] = torch.zeros((BATCH_SIZE, GENE_SIZE*GENE_N), dtype=torch.float, device=DEVICE)
  perfect_players = Players.from_params(perfect_params, bs=BATCH_SIZE, device=DEVICE)
  perfect_players.perfect = torch.ones(BATCH_SIZE, device=DEVICE, dtype=torch.bool)

  players = Players(params)
  players.credits = torch.ones((BATCH_SIZE*2,), device=DEVICE) * INIT_CREDS


  import time
  import tqdm
  pbar = tqdm.tqdm(range(5000000))
  a_players, b_players = swizzle_players(players, bs=BATCH_SIZE)

  for step in pbar:
    
    t1 = time.time()
    a_wins = torch.zeros((BATCH_SIZE,), dtype=torch.int8, device=DEVICE)
    b_wins = torch.zeros((BATCH_SIZE,), dtype=torch.int8, device=DEVICE)

    games = Games(bs=BATCH_SIZE)
    play_games(games, a_players, b_players)
    a_wins = (games.winners == PLAYERS.X)
    b_wins = (games.winners == PLAYERS.O)

    t2 = time.time()
    concat_players = Players(concat_params(a_players.params, b_players.params))
    concat_players.credits = torch.cat([a_players.credits, b_players.credits])
    if step % 100 == 0:
      print(f'mean a_player credits: {a_players.credits.mean():2f} and mean b_player credits: {b_players.credits.mean():.2f}')

    #concat_players.credits += INIT_CREDS - concat_players.credits.mean()
    t3 = time.time()
    concat_players.mate()
    mut_rate = concat_players.avg_mutation()
    trans_mut_rate = concat_players.avg_trans_mutation()
    a_players, b_players = swizzle_players(concat_players, bs=BATCH_SIZE)
    t4 = time.time()
    if step % 100 == 0:
      games_val = Games(bs=BATCH_SIZE)
      play_games(games_val, a_players, b_players, test=True)
      
      writer.add_scalar('total_moves_val', games_val.total_moves, step)
      assert ((games.illegal_movers != games.winners) | (games._total_moves == 9)).all()
      writer.add_scalar('o_illegal_move_rate', (games.illegal_movers == PLAYERS.O).sum()/BATCH_SIZE, step)
      writer.add_scalar('x_illegal_move_rate', (games.illegal_movers == PLAYERS.X).sum()/BATCH_SIZE, step)
      writer.add_scalar('o_win_rate', ((games.winners == PLAYERS.O) & (games.illegal_movers != PLAYERS.X)).sum()/BATCH_SIZE, step)
      writer.add_scalar('x_win_rate', ((games.winners == PLAYERS.X) & (games.illegal_movers != PLAYERS.O)).sum()/BATCH_SIZE, step)

      print(f'Average total moves: {games.total_moves:.2f}')
      print(f'Avg softmax scale {(torch.sigmoid(a_players.params["fake_mutation"].sum(dim=1)/10)*20).mean().item()}')
      writer.add_scalar('total_moves', games.total_moves, step)
      writer.add_scalar('avg_log_mutuation', mut_rate, step)
      writer.add_scalar('avg_trans_log_mutuation', trans_mut_rate, step)
      writer.add_scalar('draw_rate',(a_wins == b_wins).float().mean(), step)
      writer.add_scalar('first_move_optimal',(games.first_move_optimal).float().mean(), step)

    if step % 100 == 0:
      string = f'swizzling took {1000*(t4-t3):.2f}ms, playing took {1000*(t2-t1):.2f}ms, mating took {1000*(t3-t2):.2f}ms'
      pbar.set_description(string)
    if step % 1000 == 0:
      print('Saving...')
      pickle.dump(a_players.params, open('organic_dna.pkl', 'wb'))
      #a_games = Games(bs=BATCH_SIZE, device=DEVICE)
      #play_games(a_games, a_players, perfect_players, test=True)
      #b_games = Games(bs=BATCH_SIZE, device=DEVICE)
      #play_games(b_games, perfect_players, b_players, test=True)
      
      #perfect_total_moves = (a_games.total_moves + b_games.total_moves)/2
      #print(f'Vs perfect player avg total moves: {perfect_total_moves:.2f}')
      #writer.add_scalar('avg_moves_vs_perfect_player', perfect_total_moves, step)

  writer.close()
  
if __name__ == '__main__':
  for i in range(2,2000):
    bs = 5000
    name = f'run_{i}'
    train_run(name=name, embed_n=EMBED_N, bs=bs)
