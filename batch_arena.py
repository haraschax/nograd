#!/usr/bin/env python
import pickle
import torch
import math
import numpy as np
torch.set_grad_enabled(False)
import torch.nn.functional as F
from helpers import PLAYERS, next_player, BOARD_SIZE, BOARD_COLS, BOARD_ROWS
from tensorboardX import SummaryWriter
from torch.nn import functional as F
#import torch_dct as dct

MAX_MOVES = 10
BATCH_SIZE = 10
INIT_CREDS = 1
EMBED_N = 16
NOISE_SIZE = 4
MUTATION_PARAMS_SIZE = 50
INPUT_DIM = (BOARD_SIZE*3 + NOISE_SIZE) * EMBED_N
OUTPUT_DIM = EMBED_N * 7
STRAIGHT_DIM = (BOARD_SIZE*3 + NOISE_SIZE) * BOARD_SIZE
BIAS_DIM = EMBED_N
GENE_MUTATION_SIZE = 10
RUN_DIM = INPUT_DIM + OUTPUT_DIM + BIAS_DIM
GENE_SIZE = RUN_DIM + 2*GENE_MUTATION_SIZE

GENE_N = 128
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
    """
    Update the game state with the given moves for Connect4.
    Each move is a column index in [0..6], where the piece will fall
    to the lowest empty row of that column (row=5 is bottom, row=0 is top).
    """

    # 1) Basic checks
    assert moves.shape == (self.bs, 7), (
        f"Expected moves of shape (batch_size, 7), got {moves.shape}"
    )
    assert len(moves) == self.boards.shape[0]
    # Pick the chosen column for each board
    move_cols = torch.argmax(moves, dim=1)  # shape: (bs,)

    # 2) Reshape the board to (bs, 6, 7) for easy row/col indexing
    board_3d = self.boards.view(self.bs, 6, 7)

    # 3) For each batch element, find the lowest empty row in the chosen column
    # Initialize row positions to -1 (meaning "no valid row found yet")
    row_positions = torch.full(
        (self.bs,), -1, dtype=torch.long, device=self.device
    )

    # We'll iterate from bottom row=5 upward to row=0
    # and record the first empty cell we find.
    for row in reversed(range(6)):  # row=5..0
        # Check which batch elements still have no row assigned (row_positions == -1)
        # and see if that cell in (row, chosen_col) is empty (PLAYERS.NONE).
        not_assigned_yet = (row_positions == -1)
        cell_is_empty = (
            board_3d[torch.arange(self.bs, device=self.device), row, move_cols] 
            == PLAYERS.NONE
        )
        # We can place the piece if it's empty and we haven't yet assigned a row
        can_place = not_assigned_yet & cell_is_empty
        row_positions[can_place] = row  # record the row where the piece will fall

    # 4) Mark illegal moves
    # A move is illegal if there's no available row (row_positions == -1)
    # for a board that is not already over (self.game_over == 0).
    illegal_moves = (row_positions == -1) & (self.game_over == 0)

    # Record which player caused the illegal move
    self.illegal_movers[illegal_moves] = (
        PLAYERS.O if player == PLAYERS.O else PLAYERS.X
    )
    # By your convention, if the active player tries an illegal move, 
    # the other player is declared winner:
    self.winners[illegal_moves] = (
        PLAYERS.O if player == PLAYERS.X else PLAYERS.X
    )

    # Update game_over flags for boards that just became finished
    self.update_game_over()

    # 5) Place the piece in legal positions
    # We only place a piece if it's not illegal and the game isn't already over.
    legal_mask = (~illegal_moves) & (self.game_over == 0)
    batch_idx = torch.arange(self.bs, device=self.device)
    # index into [batch, row, col]
    board_3d[batch_idx[legal_mask], row_positions[legal_mask], move_cols[legal_mask]] = player

    # 6) Flatten the board back to (bs, 42)
    self.boards = board_3d.view(self.bs, 42)

    # 7) Check for winners (four-in-a-row)
    self.check_winners(player_dict)
    self.update_game_over()

  def check_winners(self, player_dict=None):
    # 1) Reshape boards to (M, 6, 7) for Connect4
    boards = self.boards.reshape((-1, 6, 7))
    M, rows, cols = boards.shape
    assert rows == 6 and cols == 7, "Each board must be 6x7 for Connect4."
    
    winners = torch.zeros(M, dtype=torch.int8, device=self.device)

    # 2) Define 2D convolution kernels for each direction
    #    Each kernel is shape (out_channels=1, in_channels=1, kernel_height, kernel_width)
    kernel_h = torch.ones((1, 1, 1, 4), device=self.device)   # horizontal (1x4)
    kernel_v = torch.ones((1, 1, 4, 1), device=self.device)   # vertical   (4x1)
    
    # Down-right diagonal: an identity matrix of size 4x4
    # [ [1, 0, 0, 0],
    #   [0, 1, 0, 0],
    #   [0, 0, 1, 0],
    #   [0, 0, 0, 1] ]
    kernel_dr = torch.eye(4, device=self.device).view(1, 1, 4, 4)
    
    # Down-left diagonal: flip the above kernel horizontally
    # [ [0, 0, 0, 1],
    #   [0, 0, 1, 0],
    #   [0, 1, 0, 0],
    #   [1, 0, 0, 0] ]
    kernel_dl = torch.flip(kernel_dr, dims=[3])  # flip over width-dimension

    for player in [PLAYERS.X, PLAYERS.O]:
        # 3) Build a binary mask for the current player: shape (M,1,6,7)
        board_mask = (boards == player).float().unsqueeze(1)

        # 4) Apply the 2D convolutions
        conv_h  = F.conv2d(board_mask, kernel_h)   # shape: (M,1,6,7-4+1)
        conv_v  = F.conv2d(board_mask, kernel_v)   # shape: (M,1,6-4+1,7)
        conv_dr = F.conv2d(board_mask, kernel_dr)  # shape: (M,1,6-4+1,7-4+1)
        conv_dl = F.conv2d(board_mask, kernel_dl)

        # 5) Check if any cell in the convolution output == 4
        #    (which means 4 consecutive 1’s, i.e. 4 in a row)
        wins_h  = (conv_h.squeeze(1)  == 4).any(dim=[1, 2])
        wins_v  = (conv_v.squeeze(1)  == 4).any(dim=[1, 2])
        wins_dr = (conv_dr.squeeze(1) == 4).any(dim=[1, 2])
        wins_dl = (conv_dl.squeeze(1) == 4).any(dim=[1, 2])

        # If any direction is True, that board has a winner of 'player'
        has_win = wins_h | wins_v | wins_dr | wins_dl
        winners[has_win] = player

    # 6) Update only boards that do not already have a winner
    no_winner_mask = (self.winners == PLAYERS.NONE)
    self.winners[no_winner_mask] = winners[no_winner_mask]

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
    noise[~self.perfect,-1] = (torch.ones_like(noise[~self.perfect,-1]) - 1.5)*2

    inputs = torch.cat([boards_onehot.reshape((-1, BOARD_SIZE*3)), noise], dim=1)

    dna_by_gene = self.params['dna'].reshape(self.bs, GENE_N, GENE_SIZE)   
    matrix_A = dna_by_gene[:,:,:INPUT_DIM].reshape((self.bs * GENE_N, BOARD_SIZE*3 + NOISE_SIZE, EMBED_N))
    bias = dna_by_gene[:,:,INPUT_DIM:INPUT_DIM+BIAS_DIM].reshape((self.bs * GENE_N, EMBED_N))
    matrix_B = dna_by_gene[:,:,INPUT_DIM+BIAS_DIM:INPUT_DIM+BIAS_DIM+OUTPUT_DIM].reshape((self.bs * GENE_N, EMBED_N, BOARD_COLS))

    embed = torch.einsum('bji, bj->bi', matrix_A, inputs.tile((1,GENE_N)).reshape((GENE_N*self.bs, BOARD_SIZE*3 + NOISE_SIZE)))
    embed = embed + bias
    embed = torch.relu(embed)
    out = torch.einsum('bji, bj->bi', matrix_B, embed)
    moves = out.reshape((self.bs, GENE_N, BOARD_COLS)).sum(dim=1)


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
    trans_mut_logit += self.params['dna'].reshape((self.bs, GENE_N, GENE_SIZE))[:,:,-2*GENE_MUTATION_SIZE:-GENE_MUTATION_SIZE].sum(dim=2).clone()

    trans_mut_rates = torch.torch.sigmoid(trans_mut_logit)/4
    mix_mutation = (torch.rand_like(self.params['dna'].reshape((self.bs, GENE_N, GENE_SIZE))[:,:,0]) < trans_mut_rates)[:,:,None].float()
    pre_mixed_params = self.params['dna'].reshape((self.bs, GENE_N, GENE_SIZE))
    new_params['dna'][can_mate] = (pre_mixed_params[can_mate]  * (1 - mix_mutation[can_mate]) + pre_mixed_params[can_mate][indices] * mix_mutation[can_mate]).reshape((-1, GENE_N*GENE_SIZE))
    '''
    '''

    for key in self.params:
      if 'mutation' in key:
        mutation_logit = self.params['mutation_mutation'].sum(dim=1)
        mutation_rate = torch.sigmoid(mutation_logit)[can_mate,None]
        param = torch.clone(self.params[key])[can_mate]
        mutation = (torch.rand_like(param) < mutation_rate).float()
        param = (1 - mutation) * param + mutation * torch.zeros_like(param).uniform_(-1, 1)
        self.params[key][dead] = param[:len(dead)]
      else:
        mutation_logit = self.params['dna_mutation'].sum(dim=1)[:,None].expand((-1, GENE_N)).clone()
        mutation_logit += self.params['dna'].reshape((self.bs, GENE_N, GENE_SIZE))[:,:,-GENE_MUTATION_SIZE:].sum(dim=2).clone()
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

  players = Players(params)
  players.credits = torch.ones((BATCH_SIZE*2,), device=DEVICE) * INIT_CREDS


  import time
  import tqdm
  pbar = tqdm.tqdm(range(200000))
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
      print(f'mean a_player credits: {a_players.credits.mean()} and mean b_player credits: {b_players.credits.mean()}')

    #concat_players.credits += INIT_CREDS - concat_players.credits.mean()
    t3 = time.time()
    concat_players.mate()
    a_players, b_players = swizzle_players(concat_players, bs=BATCH_SIZE)
    t4 = time.time()
    if step % 100 == 0:
      writer.add_scalar('total_moves_val', games.total_moves, step)
      assert ((games.illegal_movers != games.winners) | (games._total_moves == BOARD_SIZE)).all()
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
      pickle.dump(a_players.params, open('organic_dna.pkl', 'wb'))

  writer.close()
  
if __name__ == '__main__':
  for i in range(0,2000):
    bs = 500
    name = f'run_{i}'
    train_run(name=name, embed_n=EMBED_N, bs=bs)
