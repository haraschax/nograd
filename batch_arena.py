#!/usr/bin/env python
import pickle
import os
import torch
import math
import random
import tqdm
import numpy as np
torch.set_grad_enabled(False)
import torch.nn.functional as F
from helpers import PLAYERS, next_player, BOARD_SIZE
from tensorboardX import SummaryWriter
from torch.nn import functional as F
#import torch_dct as dct
from einops import rearrange

MAX_MOVES = 10
BATCH_SIZE = 10
INIT_CREDS = 0
EMBED_N = 128
NOISE_SIZE = 4
MUTATION_PARAMS_SIZE = 100
INPUT_DIM = INPUT_DIM = 32
OUTPUT_DIM = EMBED_N * BOARD_SIZE
STRAIGHT_DIM = (BOARD_SIZE*3 + NOISE_SIZE) * BOARD_SIZE
BIAS_DIM = EMBED_N
GENE_MUTATION_SIZE = 0
CORE_SIZE = 1
GENE_I = 128
GENE_J = 4
GENE_N = GENE_I * GENE_J
STATE_SIZE = 128
BOOLS_SIZE = 3
PROTEIN_N = 8
GENE_SIZE = PROTEIN_N * 3 + 3
LAYERS= 2

DNA_SIZE = GENE_N * GENE_SIZE
OFFSPRING = 2
GAMES_PER_MATE = 3
 

DNA_SIZE = GENE_N * GENE_SIZE

DEVICE = 'cuda'

def quantize(x, N):
  x = (x + 1.0) / 2
  return torch.floor(x * N).long()


def is_winner(board, player):
    return (np.any(np.all(board == player, axis=1)) or
            np.any(np.all(board == player, axis=0)) or
            np.all(np.diag(board) == player) or
            np.all(np.diag(np.fliplr(board)) == player))

def is_draw(board):
    return not np.any(board == 0)

def generate_valid_boards(current_board, player, all_boards, board_hashes):
    board_hash = current_board.tobytes()
    if board_hash in board_hashes:
        return
    #if is_winner(current_board, PLAYERS.X) or is_winner(current_board, PLAYERS.O) or is_draw(current_board):
    #    return
    board_hashes.add(board_hash)
    all_boards.append(current_board.copy())
    for i in range(3):
        for j in range(3):
            if current_board[i, j] == 0:
                current_board[i, j] = player
                generate_valid_boards(current_board, next_player(player), all_boards, board_hashes)
                current_board[i, j] = 0

def get_all_valid_boards():
    all_boards = []
    board_hashes = set()
    initial_board = np.zeros((3, 3), dtype=int)
    generate_valid_boards(initial_board, PLAYERS.X, all_boards, board_hashes)
    return all_boards

def get_optimal_move(board, player):
    #assert not is_winner(board, PLAYERS.X)
    #assert not is_winner(board, PLAYERS.O)
    #assert not is_draw(board)
    scores = np.zeros((3, 3), dtype=int)
    scores[board != PLAYERS.NONE] = -1
    if is_winner(board, player):
      return scores, 1
    if is_draw(board):
      return scores, 0
    if is_winner(board, next_player(player)):
      return scores, -1
    for i in range(3):
      for j in range(3):
        if board[i, j] == PLAYERS.NONE:
          board[i, j] = player
          if is_winner(board, player):
            score = 1
          elif is_draw(board):
            score = 0
          else:
            _, score = get_optimal_move(board, next_player(player))
            score = -score
          scores[i, j] = score
          board[i, j] = PLAYERS.NONE
    best_score = np.max(scores)
    return scores, best_score

def unique_int_from_board(board):
    return int(np.sum(board.flatten() * (3 ** np.arange(9))))

def unique_int_from_board_torch(board):
  return (board.reshape((-1,9)) * (3 ** torch.arange(9, device=board.device)).reshape((1,9))).to(dtype=torch.int64).sum(dim=1)

def generate_perfect_moves():
    all_boards = get_all_valid_boards()
    board_move_pairs = []
    for board in tqdm.tqdm(all_boards):
        player = PLAYERS.O if np.sum(board == PLAYERS.X) > np.sum(board == PLAYERS.O) else PLAYERS.X
        good_moves, score = get_optimal_move(board, player)
        board_move_pairs.append((board, good_moves, player, score))
    full_board_scores = np.nan*np.zeros((20000,), dtype=int)
    full_board_players = np.nan*np.zeros((20000,), dtype=int)
    full_board_moves = np.zeros((20000,9), dtype=int)
    for board, good_moves, player, score in board_move_pairs:
      board_hash = unique_int_from_board(board)
      full_board_scores[board_hash] = score
      full_board_players[board_hash] = player
      full_board_moves[board_hash] = good_moves.flatten().astype(int)
    board_dict = {'scores': full_board_scores, 'players': full_board_players, 'moves': full_board_moves}
    #board_dict = {str(k): (k, move, player, score) for k,move,player,score in board_move_pairs}
    pickle.dump(board_dict, open('perfect_moves.pkl', 'wb'))
    return board_dict

def get_losing_move_ratio(player_instance):
  if os.path.isfile('perfect_moves.pkl'):
    perfect_dataset = pickle.load(open('perfect_moves.pkl', 'rb'))
  else:
    perfect_dataset = generate_perfect_moves()
  test_player = Players(splice_params(player_instance.params, [0]))
  losing_moves = 0
  total = 0
  for board in tqdm.tqdm(get_all_valid_boards()):
    if is_winner(board, PLAYERS.X) or is_winner(board, PLAYERS.O) or is_draw(board):
      continue
    current_player, score_before = perfect_dataset['players'][unique_int_from_board(board)], perfect_dataset['scores'][unique_int_from_board(board)]
    board_tensor = torch.tensor(board.flatten(), dtype=torch.int64, device=DEVICE).unsqueeze(0)
    model_move_probs = test_player.play(board_tensor, test=True, current_player=current_player)
    move_index = torch.argmax(model_move_probs, dim=1).item()
    row, col = move_index // 3, move_index % 3

    board_after = board.copy()
    if board_after[row, col] == PLAYERS.NONE:
      board_after[row, col] = current_player
      if is_winner(board_after, current_player):
        score_after = 1
      elif is_draw(board_after):
        score_after = 0
      else:
        s = perfect_dataset['scores'][unique_int_from_board(board_after)]
        score_after = -s
    else:
      score_after = -1
    
    if score_after < score_before:
        losing_moves += 1
    total += 1
  return losing_moves / total

class Games():
  def __init__(self, bs=BATCH_SIZE, device=DEVICE, perfect_dataset=None):
    self.bs = bs
    self.device = device
    self.boards = torch.zeros((self.bs, BOARD_SIZE), dtype=torch.int8, device=self.device)
    self.winners = torch.zeros((self.bs,), dtype=torch.int8, device=self.device)
    self.illegal_movers = torch.zeros((self.bs,), dtype=torch.int8, device=self.device)
    self.update_game_over()
    self.perfect_dataset = perfect_dataset
    if perfect_dataset is None:
      self.perfect_dataset = pickle.load(open('perfect_moves.pkl', 'rb'))
    else:
      self.perfect_dataset = perfect_dataset
    if self.perfect_dataset is not None:
      self.perfect_scores = torch.tensor(self.perfect_dataset['scores'], device=self.device)
      

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
    self.check_winners(player_dict, player, test=test)
    self.update_game_over()

  def check_winners(self, player_dict, current_player, test):
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

    if False and not test:
      unique_ints = unique_int_from_board_torch(boards)
      scores = self.perfect_scores[unique_ints]
      filt =  ((self.winners == PLAYERS.NONE) & ((self.boards == PLAYERS.NONE).sum(dim=1) > 0))
      self.winners[filt & (scores > 0)] = next_player(current_player)
      assert torch.all(scores[filt] >= 0)

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
    self.params = params

    if os.path.isfile('perfect_moves.pkl'):
      self.perfect_dataset = pickle.load(open('perfect_moves.pkl', 'rb'))
    else:
      self.perfect_dataset = generate_perfect_moves()
    self.perfect_move_tensor_raw = 1e8*torch.tensor(self.perfect_dataset['moves'], dtype=torch.float, device=DEVICE)[:3**9]
    self.set_perfect_move_ratio(0.95)

  def set_perfect_move_ratio(self, ratio):
    self.perfect_move_tensor = self.perfect_move_tensor_raw.clone()
    idx = int((1- ratio) * self.perfect_move_tensor.shape[0])
    self.perfect_move_tensor[:idx] = 0

  def run_dna(self, dna_by_gene, input_vector):
    dna_by_gene = dna_by_gene.reshape((self.bs * GENE_I, GENE_J, GENE_SIZE))
    input_vector_clone = input_vector.clone()
    device = input_vector.device
    B = self.bs * GENE_I
    batch_idx = torch.arange(B, device=device)//GENE_I

    for i in range(dna_by_gene.shape[1]):
      write_val = torch.ones((self.bs * GENE_I), device=input_vector.device, dtype=torch.float)
      for l in range(PROTEIN_N):
        gidx = l*3
        idx_in = quantize(dna_by_gene[:, i, gidx], STATE_SIZE)
        val = dna_by_gene[:, i, gidx + 1]
        mask = (dna_by_gene[:, i, gidx + 2].reshape((self.bs, GENE_I)) > 0).flatten()
        write_val = write_val + (mask * val * input_vector_clone[batch_idx, idx_in])
      val_out = dna_by_gene[:, i, -2]#.reshape((self.bs, -1))
      idx_out = quantize(dna_by_gene[:, i, -1], STATE_SIZE)
      update = torch.zeros_like(input_vector_clone)
      update[batch_idx, idx_out] = val_out * torch.relu(write_val + dna_by_gene[:, i, -3])
      input_vector_clone = input_vector_clone + torch.tanh(update)
    return input_vector_clone

  '''
  def run_dna(self, dna_by_gene, input_vector):
    #input_vector = input_vector.clone().bool() 
    input_vector_clone = input_vector.clone()
    dna = rearrange(dna_by_gene, 'b (i j)-> b i j', i=LAYERS)

    #scales = self.scale_mutation
    for i in range(LAYERS):
      #input_vector_clone = torch.nn.functional.layer_norm(input_vector_clone, (STATE_SIZE,))
      A = dna[:,i, :STATE_SIZE*STATE_SIZE].reshape((-1, STATE_SIZE, STATE_SIZE))
      b = dna[:,i, STATE_SIZE*STATE_SIZE:STATE_SIZE*STATE_SIZE + STATE_SIZE]
      C = dna[:,i, STATE_SIZE*STATE_SIZE + STATE_SIZE:2*STATE_SIZE*STATE_SIZE + STATE_SIZE].reshape((-1, STATE_SIZE, STATE_SIZE))
      d = dna[:,i, 2*STATE_SIZE*STATE_SIZE + STATE_SIZE:2*STATE_SIZE*STATE_SIZE + 2*STATE_SIZE]

      x = torch.einsum('bji, bj->bi', A, input_vector_clone)
      #x = x*scales[:,i][:,None] + b*scales[:,LAYERS+i][:,None]
      x = x + b
      x = torch.relu(x)
      x = torch.einsum('bji, bj->bi', C, x)
      x = x + d
      x = torch.nn.functional.layer_norm(x, (STATE_SIZE,))
      input_vector_clone += x

    return input_vector_clone
  '''



  @property
  def mutation_mutation(self):
    mut_mut_exp = torch.tanh(self.params['mutation_mutation'].sum(dim=1)/20)
    return 10**(-10*mut_mut_exp)
  
  @property
  def full_gene_mutation(self):
    mut_mut_exp = torch.tanh(self.params['full_gene_mutation'].sum(dim=1)/20)
    return 10**(-10*mut_mut_exp)
    #return 1e-3 * torch.ones_like(mut_mut_exp)

  @property
  def mutation(self):
    mut_exp = torch.tanh(self.params['mutation'].sum(dim=1)/20)
    return 10**(-10*mut_exp)
  
  @property
  def credits(self):
    return self.params['credits']

  @property
  def trans_mutation(self):
    trans_mut_exp = torch.tanh(self.params['trans_mutation'].sum(dim=1)/20)
    trans_mutation = 10**(-7*trans_mut_exp -1)
    return torch.clamp(trans_mutation, 0, 0.1)

  @property
  def output_scale_mutation(self):
    scales_exp = torch.tanh(self.params['output_scale_mutation'].sum(dim=1)/20)
    scales = 10**(5*scales_exp)
    return scales

  @property
  def switch_gene_prob(self):
    return torch.sigmoid(self.params['switch_gene_mutation'].sum(dim=1)/20)

  def play(self, boards, test=False, current_player=PLAYERS.X):
    unique_ints = unique_int_from_board_torch(boards)
    boards_onehot_raw_full = F.one_hot(unique_ints, num_classes=(3**9)).float()

    boards_recode = torch.zeros_like(boards)
    boards_recode[boards == current_player] = 1
    boards_recode[boards == next_player(current_player)] = 2
    boards_onehot_raw = F.one_hot(boards_recode.long(), num_classes=3).float().reshape((boards.size(0), -1))

    #A = self.params['dna'][:,:(3**9)*9].reshape((-1, 3**9, 9))
    #moves = torch.einsum('bji, bj->bi', A, boards_onehot_raw)
    state = torch.zeros((boards.size(0), STATE_SIZE), dtype=torch.float, device=boards.device)
    state[:,:boards_onehot_raw.size(1)] = boards_onehot_raw
    state[:,boards_onehot_raw.size(1):INPUT_DIM] = 1.0 if current_player == PLAYERS.X else -1.0
    state[:,INPUT_DIM:INPUT_DIM+32] = 1.0
    state[:,INPUT_DIM+32:-9] = torch.rand((boards.size(0), 64-9))

    state = self.run_dna(self.params['dna'], state)
    moves = state[:, -9:]
    moves = self.output_scale_mutation[:,None] * moves

    perfect_moves = torch.einsum('ji, bj->bi', self.perfect_move_tensor, boards_onehot_raw_full)
    moves += perfect_moves

    move_probs = torch.softmax(moves, dim=1)
    sampled_indices = torch.multinomial(move_probs, num_samples=1)
    moves = F.one_hot(sampled_indices.squeeze(-1), num_classes=moves.size(1)).float()
    
    if not test:
      moves[boards == PLAYERS.NONE] += 1e8 * torch.ones_like(moves[boards == PLAYERS.NONE]) * (torch.rand_like(moves[boards == PLAYERS.NONE]) < 0.001).float()
    

    #moves[boards != PLAYERS.NONE] -= 1e8 * torch.ones_like(moves[boards != PLAYERS.NONE])# * (torch.rand_like(moves[boards == PLAYERS.NONE]) < 0.1).float()
    '''
    for i, board in enumerate(boards):
      board_np = board.cpu().numpy().reshape((3,3))
      max_score = -10
      move_idx = 0
      move_tuples = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
      np.random.shuffle(move_tuples)
      for j,k in move_tuples:
        if board_np[j,k] == PLAYERS.NONE:
          board_after = board_np.copy()
          board_after[j,k] = current_player
          score = -self.perfect_dataset['scores'][unique_int_from_board(board_after)]
          if np.isnan(score):
            score = -5
          if score > max_score:
            max_score = score
            move_idx = j * 3 + k
      moves[i,move_idx] += 1e8
    '''
    return moves

  def avg_trans_mutation(self):
    return math.log(self.trans_mutation.mean().float().item())

  def avg_mutation(self):
    return math.log(self.mutation.mean().float().item())

  def mate(self, init_credits=INIT_CREDS):

    cred = self.credits
    bs = len(cred)
    can_mate = torch.argsort(cred, descending=True)
    can_mate = can_mate[:int(len(can_mate) /OFFSPRING)]
    cred *= 0

    repro_params = {}
    for key in self.params:
      if 'credits' in key:
        continue
      repro_params[key] = self.params[key][can_mate,None,:].repeat(1, OFFSPRING, 1).reshape((bs, self.params[key].shape[1]))
    mutation_mask = torch.ones((len(can_mate), OFFSPRING), dtype=torch.float, device=DEVICE)
    #mutation_mask[:,0] = 0.0
    mutation_mask = mutation_mask.reshape((bs))


    self.params = repro_params
    repro_params['credits'] = cred

    indices = torch.randperm(bs)
    gene_indices = torch.randperm(GENE_N)
    trans_mut_rates = self.trans_mutation[:,None].clone()
    for key in self.params:
      if 'credits' in key:
        continue
      if key == 'dna':
        pre_mixed_params = self.params[key].reshape((bs, GENE_N, GENE_SIZE))
        mix_mutation = (torch.rand_like(pre_mixed_params) < trans_mut_rates[:,:,None]).float()[:,:,0]
        mix_mutation *= mutation_mask[:,None]
        mix_mutation = mix_mutation[:,:,None]
        mixed_params = pre_mixed_params[indices]
        switch_gene = (torch.rand_like(self.switch_gene_prob) < self.switch_gene_prob).float()[:,None,None]
        mixed_params = switch_gene * mixed_params[:, gene_indices] + (1 - switch_gene) * mixed_params
        self.params[key] = (pre_mixed_params  * (1 - mix_mutation) + mixed_params * mix_mutation).reshape((bs, GENE_N*GENE_SIZE))
      else:
        mut_indices = torch.randperm(MUTATION_PARAMS_SIZE)
        mix_mutation = (torch.rand_like(self.params[key]) < trans_mut_rates).float()
        mix_mutation *= mutation_mask[:,None]
        pre_mixed_params = self.params[key]
        self.params[key] = (pre_mixed_params  * (1 - mix_mutation) + pre_mixed_params[indices][:,mut_indices] * mix_mutation)

    for key in self.params:
      if 'credits' in key:
        continue
      if 'mutation' in key:
        mutation_rate = self.mutation_mutation.clone()[:,None]
        param = torch.clone(self.params[key])
        mutation = mutation_mask[:,None]*(torch.rand_like(param) < mutation_rate).float()
        self.params[key] = (1 - mutation) * param + mutation * torch.zeros_like(param).uniform_(-1, 1)
      else:
        mutation_rate = self.mutation[:,None,None]
        #mutation_rate = mutation_rate# * (10**(5*self.params['dna'][:,(3**9)*9:])).repeat((1,2))
        param = torch.clone(self.params[key]).reshape((-1, GENE_N, GENE_SIZE))
        mutation = mutation_mask[:,None,None]*(torch.rand_like(param)[:,:,:] < mutation_rate).float()
        full_mutation = mutation_mask[:,None,None] * (torch.rand_like(param)[:,:,0:1] < self.full_gene_mutation[:,None,None])
        param = (1- mutation) * param + mutation * torch.zeros_like(param).uniform_(-1, 1)
        param = (1- full_mutation) * param + full_mutation * torch.zeros_like(param).uniform_(-1, 1)
        self.params[key] = param.reshape((-1, GENE_N*GENE_SIZE))
    #self.params['dna'][:,:] = 1e8*self.perfect_dna.reshape((-1,))[10000:]

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
      player_dict[player].credits[(games.winners == player)] += 1.0
      player_dict[player].credits[games.losers == player] -= 1.0

def splice_params(params, indices):
  new_params = {}
  for key in params:
    new_params[key] = torch.clone(params[key][indices])
  return new_params

def concat_params(params1, params2, slc1=slice(0,None), slc2=slice(0,None)):
  new_params = {}
  for key in params1:
    new_params[key] = torch.cat([params1[key][slc1], params2[key][slc2]])
  return new_params

def swizzle_players(players):
  bs = players.params['dna'].shape[0] // 2
  indices = torch.randperm(bs*2)
  x_players = Players(splice_params(players.params, indices[:bs]))
  o_players = Players(splice_params(players.params, indices[bs:]))
  return x_players, o_players

def concat_players(a_players, b_players):
  players = Players(concat_params(a_players.params, b_players.params))
  return players

def write_metrics(step, writer, games, a_players, b_players):
  bs = a_players.params['dna'].shape[0]
  players = concat_players(a_players, b_players)
  a_players, b_players = swizzle_players(players)
  a_players.set_perfect_move_ratio(1.0)
  games_val = Games(bs=bs)
  play_games(games_val, a_players, b_players, test=True)
  players = concat_players(a_players, b_players)
  a_players, b_players = swizzle_players(players)
  b_players.set_perfect_move_ratio(1.0)
  games_val2 = Games(bs=bs)
  play_games(games_val2, a_players, b_players, test=True)

  writer.add_scalar('total_moves_val', torch.mean(torch.stack([games_val.total_moves, games_val2.total_moves])), step)
  assert ((games.illegal_movers != games.winners) | (games._total_moves == 9)).all()
  writer.add_scalar('o_illegal_move_rate', (games.illegal_movers == PLAYERS.O).sum()/bs, step)
  writer.add_scalar('x_illegal_move_rate', (games.illegal_movers == PLAYERS.X).sum()/bs, step)
  writer.add_scalar('o_win_rate', ((games.winners == PLAYERS.O) & (games.illegal_movers != PLAYERS.X)).sum()/bs, step)
  writer.add_scalar('x_win_rate', ((games.winners == PLAYERS.X) & (games.illegal_movers != PLAYERS.O)).sum()/bs, step)
  writer.add_scalar('total_moves', games.total_moves, step)
  writer.add_scalar('draw_rate',(games.winners == PLAYERS.NONE).float().mean(), step)
  writer.add_scalar('mutation', a_players.mutation.mean(), step)
  writer.add_scalar('mutation_mutation', a_players.mutation_mutation.mean(), step)
  writer.add_scalar('full_gene_mutation', a_players.full_gene_mutation.mean(), step)
  writer.add_scalar('trans_mutation', a_players.trans_mutation.mean(), step)
  writer.add_scalar('output_scale_mutation', a_players.output_scale_mutation.mean(), step)
  writer.add_scalar('switch_gene_mutation', a_players.switch_gene_prob.mean(), step)


def init_players(bs=BATCH_SIZE):
  params = {}
  params['dna'] = torch.zeros((bs*2, GENE_N*GENE_SIZE), dtype=torch.float, device=DEVICE).uniform_(-1, 1)
  #params['dna'] = torch.zeros((bs*2, (3**9)*9), dtype=torch.float, device=DEVICE).uniform_(-1, 1)

  params['mutation'] = torch.zeros((bs*2, MUTATION_PARAMS_SIZE), dtype=torch.float, device=DEVICE).uniform_(-1, 1)
  params['trans_mutation'] = torch.zeros((bs*2, MUTATION_PARAMS_SIZE), dtype=torch.float, device=DEVICE).uniform_(-1, 1)
  params['mutation_mutation'] = torch.zeros((bs*2, MUTATION_PARAMS_SIZE), dtype=torch.float, device=DEVICE).uniform_(-1, 1)
  params['output_scale_mutation'] = torch.zeros((bs*2, MUTATION_PARAMS_SIZE), dtype=torch.float, device=DEVICE).uniform_(-1, 1)
  params['switch_gene_mutation'] = torch.zeros((bs*2, MUTATION_PARAMS_SIZE), dtype=torch.float, device=DEVICE).uniform_(-1, 1)
  params['full_gene_mutation'] = torch.zeros((bs*2, MUTATION_PARAMS_SIZE), dtype=torch.float, device=DEVICE).uniform_(-1, 1)
  params['credits'] = torch.zeros((bs*2,), dtype=torch.float, device=DEVICE)
  players = Players(params)
  return players

def train_run(name='', bs=BATCH_SIZE):

  writer = SummaryWriter(f'runs/{name}')
  players = init_players(bs=bs)

  pbar = tqdm.tqdm(range(50_000))

  for step in pbar:
    for _ in range(GAMES_PER_MATE):
      a_players, b_players = swizzle_players(players)
      games = Games(bs=bs)
      play_games(games, a_players, b_players)
      players = concat_players(a_players, b_players)
      a_players, b_players = swizzle_players(players)
      players = concat_players(a_players, b_players)

    players.mate()

    pbar.set_description(f'Average total moves: {games.total_moves:.2f}')

    if step % 100 == 0:
      write_metrics(step, writer, games, a_players, b_players)
      if step % 1000 == 0 and step > 0:
        pickle.dump(a_players.params, open('organic_dna.pkl', 'wb'))
        losing_move_ratio = get_losing_move_ratio(a_players)
        writer.add_scalar('losing_move_ratio', losing_move_ratio, step)
        print(f'Losing move ratio at step {step}: {losing_move_ratio:.2f} bad move ratio')
  writer.close()


if __name__ == '__main__':
  for i in range(600,100000):
    bs = 5000
    name = f'run_{i}'
    train_run(name=name, bs=bs)
