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
GENE_J = 5
GENE_N = GENE_I * GENE_J
STATE_SIZE = 32
BOOLS_SIZE = 3
LAYERS = GENE_J
GENE_CORE_SIZE = 4
GENE_SIZE = 4

DNA_SIZE = GENE_N * GENE_SIZE
OFFSPRING = 2
GAMES_PER_MATE = 1

DEVICE = 'cuda'
LIFE_SIZE = STATE_SIZE
#INDICES = torch.randperm(LIFE_SIZE)
INDICES = range(LIFE_SIZE//BOARD_SIZE, LIFE_SIZE-2, LIFE_SIZE//BOARD_SIZE)

FIXED_SEED_LIFE  = (torch.randn((LIFE_SIZE, LIFE_SIZE), device=DEVICE) > 0.0).bool()


def life_step_conway(board: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], device=board.device, dtype=torch.float32)

    padded = F.pad(board.float().unsqueeze(1),  (1, 1, 1, 1), mode='circular')
    neighbours = F.conv2d(padded, kernel[None, None])[:,0]
    alive       = board.bool()
    next_board  = (neighbours == 3) | (alive & (neighbours == 2))
    return next_board.to(board.dtype)

def life_step_brians(board: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], device=board.device, dtype=torch.float32)

    padded     = F.pad((board == 1).float().unsqueeze(1), (1, 1, 1, 1), mode='circular')
    neighbours = F.conv2d(padded, kernel[None, None])[:, 0]
    next_board = torch.zeros_like(board)

    # Rule 1: ON → DYING
    next_board[board == 1] = 2

    # Rule 2: OFF with exactly two ON neighbours → ON
    birth_mask = (board == 0) & (neighbours == 2)
    next_board[birth_mask] = 1

    # Rule 3 (DYING → OFF) is already satisfied by the default 0
    return next_board


def life_step(board: torch.Tensor) -> torch.Tensor:
    """
    Seeds cellular automaton.
    Rule: a cell is born (1) iff it is currently 0 and has exactly two live neighbours;
          all live cells die every step.
    States: 0 = dead, 1 = live.
    """
    kernel = torch.tensor([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], device=board.device, dtype=torch.float32)

    # count live neighbours with circular padding
    padded = F.pad(board.float().unsqueeze(1), (1, 1, 1, 1), mode='circular')
    neighbours = F.conv2d(padded, kernel[None, None])[:, 0]

    # birth if exactly two neighbours; no survival
    next_board = ((neighbours == 2) & (~board.bool())).to(board.dtype)
    return next_board

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
    if is_winner(board, player):
      return None, 1
    if is_draw(board):
      return None, 0
    if is_winner(board, next_player(player)):
      return None, -1
    best_score = -np.inf
    best_move = None
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
                if score > best_score:
                    best_score = score
                    best_move = (i, j)
                board[i, j] = PLAYERS.NONE
    return best_move, best_score

def unique_int_from_board(board):
    return int(np.sum(board.flatten() * (3 ** np.arange(BOARD_SIZE))))

def unique_int_from_board_torch(board):
  return (board.reshape((-1,BOARD_SIZE)) * (3 ** torch.arange(BOARD_SIZE, device=board.device)).reshape((1,BOARD_SIZE))).to(dtype=torch.int64).sum(dim=1)

def generate_perfect_moves():
    all_boards = get_all_valid_boards()
    board_move_pairs = []
    for board in tqdm.tqdm(all_boards):
        player = PLAYERS.O if np.sum(board == PLAYERS.X) > np.sum(board == PLAYERS.O) else PLAYERS.X
        move, score = get_optimal_move(board, player)
        board_move_pairs.append((board, move, player, score))
    full_board_scores = np.nan*np.zeros((20000,), dtype=int)
    full_board_players = np.nan*np.zeros((20000,), dtype=int)
    for board, move, player, score in board_move_pairs:
      board_hash = unique_int_from_board(board)
      full_board_scores[board_hash] = score
      full_board_players[board_hash] = player
    board_dict = {'scores': full_board_scores, 'players': full_board_players}
    return board_dict

def get_losing_move_ratio(player_instance):
  if os.path.isfile('perfect_moves.pkl'):
    perfect_dataset = pickle.load(open('perfect_moves.pkl', 'rb'))
  else:
    perfect_dataset = generate_perfect_moves()
    pickle.dump(perfect_dataset, open('perfect_moves.pkl', 'wb'))
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


def check_winner(board, conv_layer):
    board_tensor = board.float().unsqueeze(1)
    conv_output = conv_layer(board_tensor).squeeze()
    return conv_output

class Games():
  def __init__(self, bs=BATCH_SIZE, device=DEVICE, perfect_dataset=None):
    self.bs = bs
    self.device = device
    self.boards = torch.zeros((self.bs, BOARD_SIZE), dtype=torch.int8, device=self.device)
    self.winners = torch.zeros((self.bs,), dtype=torch.int8, device=self.device)
    self.illegal_movers = torch.zeros((self.bs,), dtype=torch.int8, device=self.device)
    self.update_game_over()
    self.perfect_dataset = perfect_dataset
    if perfect_dataset is not None:
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
    self.weights = None

    if os.path.isfile('perfect_moves.pkl'):
      self.perfect_dataset = pickle.load(open('perfect_moves.pkl', 'rb'))
    else:
      self.perfect_dataset = generate_perfect_moves()


  def run_dna(self, dna_by_gene, input_vector):
    dna = rearrange(dna_by_gene, 'b (i j k)-> b i j k', i=GENE_I, k=GENE_SIZE)
    input_vector_clone = input_vector.clone()

    life_board = torch.zeros((self.bs, LIFE_SIZE, LIFE_SIZE), device=self.device, dtype=torch.bool)
    life_board += FIXED_SEED_LIFE[None, :, :].bool()

    B, I, J, K = dna.shape
    device   = dna.device
    half     = LIFE_SIZE / 2
    for j in range(J):
      life_board = life_step(life_board)

      idxs  = ((dna[:,:,j,:3] + 1) * half).long() 
      
      in_vals  = (input_vector_clone.gather(1, idxs[...,0])).bool()
      batch_idx = torch.arange(B, device=device).view(-1, 1).expand(-1, I) 

      life_board[batch_idx.reshape(-1),
                 idxs[...,1].reshape(-1),
                 idxs[...,2].reshape(-1)] = in_vals.reshape(-1)
     
    input_vector_clone = life_board.sum(dim=1)
    return input_vector_clone

  @property
  def mutation_mutation(self):
    mut_mut_exp = torch.sigmoid(self.params['mutation_mutation'].sum(dim=1)/20)
    return 10**(-10*mut_mut_exp)

  @property
  def mutation(self):
    mut_exp = torch.sigmoid(self.params['mutation'].sum(dim=1)/20)
    return 10**(-10*mut_exp)
  
  @property
  def trans_mutation(self):
    trans_mut_exp = torch.sigmoid(self.params['trans_mutation'].sum(dim=1)/20)
    trans_mutation = 10**(-10*trans_mut_exp - 1)
    #trans_mutation[:] = 0.0
    return trans_mutation

  @property
  def scale_mutation(self):
    scales_exp = torch.tanh(self.params['scales_mutation'].reshape((-1, MUTATION_PARAMS_SIZE, LAYERS*2)).sum(dim=1))
    scales = 10**(5*scales_exp)
    return scales

  def play(self, boards, test=False, current_player=PLAYERS.X):
    boards_onehot_raw = F.one_hot(boards.long(), num_classes=3)
    boards_onehot = boards_onehot_raw.clone()
    

    state = torch.zeros((self.bs, STATE_SIZE), device=boards.device)
    state[:,:BOARD_SIZE*3] = boards_onehot.reshape((-1, BOARD_SIZE*3))
    if current_player == PLAYERS.X:
      state[:,BOARD_SIZE*3] = 1.0
    state[:,BOARD_SIZE*3 + 1] = 1.0
    state = self.run_dna(self.params['dna'], state)

    moves = torch.clone(state[:,INDICES]).float()

    if not test:
      moves[boards == PLAYERS.NONE] += 1e8 * torch.ones_like(moves[boards == PLAYERS.NONE]) * (torch.rand_like(moves[boards == PLAYERS.NONE]) < 0.01).float()
    return moves

  def mate(self,):
    cred = self.params['credits']
    bs = len(cred)
    can_mate = torch.argsort(cred, descending=True)
    #print(torch.max(cred), torch.min(cred), len(can_mate), torch.mean(abs(cred).float()))
    #print(self.params['credits'][can_mate][:10], self.params['x_games'][can_mate][:10])
    #print(self.credits.max().item(), self.credits.min().item(), (self.credits == self.credits.max()).sum().item(), (self.credits == self.credits.min()).sum().item())
    self.params['credits'][:] = 0
    self.params['x_games'][:] = 0
    can_mate = can_mate[:int(len(can_mate) /OFFSPRING)]


    repro_params = {}
    for key in self.params:
      if key in ['credits', 'x_games']:
        continue
      repro_params[key] = self.params[key][can_mate,None,:].repeat(1, OFFSPRING, 1).reshape((bs, self.params[key].shape[1]))
    repro_params['credits'] = torch.zeros((bs*2), dtype=torch.long, device=DEVICE)
    repro_params['x_games'] = torch.zeros((bs*2), dtype=torch.long, device=DEVICE)
    

    self.params = repro_params

    indices = torch.randperm(bs)
    trans_mut_rates = self.trans_mutation[:,None].clone()
    for key in self.params:
      if key in ['credits', 'x_games']:
        continue
      if 'mutation' in key:
        continue
      mix_mutation = (torch.rand_like(self.params[key]) < trans_mut_rates).float().reshape((-1, GENE_N, GENE_SIZE))[:,:,:1]
      pre_mixed_params = self.params[key].reshape((-1, GENE_N, GENE_SIZE))
      self.params[key] = (pre_mixed_params  * (1 - mix_mutation) + pre_mixed_params[indices] * mix_mutation).reshape((-1, GENE_N*GENE_SIZE))

    for key in self.params:
      if key in ['credits', 'x_games']:
        continue
      if 'mutation' in key:
        mutation_rate = self.mutation_mutation.clone()[:,None]
        param = torch.clone(self.params[key])
        mutation = (torch.rand_like(param) < mutation_rate).float()
        param = (1 - mutation) * param + mutation * torch.zeros_like(param).uniform_(-1, 1)
        self.params[key] = param
      else:
        mutation_rate = self.mutation.clone()[:,None]
        param = torch.clone(self.params[key])
        mutation = (torch.rand_like(param) < mutation_rate).float()
        param = (1 - mutation) * param + mutation * torch.zeros_like(param).uniform_(-1, 1)
        self.params[key] = param
      

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
      player_dict[player].params['credits'][(games.winners == player)] += 1
      player_dict[player].params['credits'][games.losers == player] -= 1

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

def swizzle_players(players):
  bs = players.params['dna'].shape[0] // 2
  indices = torch.randperm(bs*2)
  x_players = Players(splice_params(players.params, indices[:bs]))
  o_players = Players(splice_params(players.params, indices[bs:]))
  x_players.params['x_games'] += 1
  return x_players, o_players

def concat_players(a_players, b_players):
  players = Players(concat_params(a_players.params, b_players.params))
  return players

def write_metrics(step, writer, games, a_players, b_players):
  bs = a_players.params['dna'].shape[0]
  games_val = Games(bs=bs)
  play_games(games_val, a_players, b_players, test=True)

  writer.add_scalar('total_moves_val', games_val.total_moves, step)
  assert ((games.illegal_movers != games.winners) | (games._total_moves == 9)).all()
  writer.add_scalar('o_illegal_move_rate', (games.illegal_movers == PLAYERS.O).sum()/bs, step)
  writer.add_scalar('x_illegal_move_rate', (games.illegal_movers == PLAYERS.X).sum()/bs, step)
  writer.add_scalar('o_win_rate', ((games.winners == PLAYERS.O) & (games.illegal_movers != PLAYERS.X)).sum()/bs, step)
  writer.add_scalar('x_win_rate', ((games.winners == PLAYERS.X) & (games.illegal_movers != PLAYERS.O)).sum()/bs, step)
  writer.add_scalar('total_moves', games.total_moves, step)
  writer.add_scalar('draw_rate',(games.winners == PLAYERS.NONE).float().mean(), step)
  writer.add_scalar('mutation', a_players.mutation.mean(), step)
  writer.add_scalar('mutation_mutation', a_players.mutation_mutation.mean(), step)
  writer.add_scalar('trans_mutation', a_players.trans_mutation.mean(), step)


def init_players(bs=BATCH_SIZE):
  params = {}
  params['dna'] = torch.zeros((bs*2, GENE_I*GENE_J*GENE_SIZE), dtype=torch.float, device=DEVICE).uniform_(-1, 1)
  params['mutation'] = torch.zeros((bs*2, MUTATION_PARAMS_SIZE), dtype=torch.float, device=DEVICE).uniform_(-1, 1)
  params['trans_mutation'] = torch.zeros((bs*2, MUTATION_PARAMS_SIZE), dtype=torch.float, device=DEVICE).uniform_(-1, 1)
  params['mutation_mutation'] = torch.zeros((bs*2, MUTATION_PARAMS_SIZE), dtype=torch.float, device=DEVICE).uniform_(-1, 1)
  params['output_scale_mutation'] = torch.zeros((bs*2, MUTATION_PARAMS_SIZE), dtype=torch.float, device=DEVICE).uniform_(-1, 1)
  params['scales_mutation'] = torch.zeros((bs*2, MUTATION_PARAMS_SIZE*LAYERS*2), dtype=torch.float, device=DEVICE).uniform_(-1, 1)
  params['credits'] = torch.zeros((bs*2), dtype=torch.long, device=DEVICE)
  params['x_games'] = torch.zeros((bs*2), dtype=torch.long, device=DEVICE)
  players = Players(params)
  return players

def train_run(name='', bs=BATCH_SIZE):

  writer = SummaryWriter(f'runs/{name}')
  players = init_players(bs=bs)

  pbar = tqdm.tqdm(range(200000))

  for step in pbar:
    for _ in range(GAMES_PER_MATE):
      a_players, b_players = swizzle_players(players)
      games = Games(bs=bs)
      play_games(games, a_players, b_players)
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
  for i in range(12,2000):
    bs = 5000
    name = f'run_{i}'
    train_run(name=name, bs=bs)
