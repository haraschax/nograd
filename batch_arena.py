#!/usr/bin/env python
import argparse
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from helpers import BOARD_SIZE, PLAYERS, next_player


WIN_LINES = (
  (0, 1, 2),
  (3, 4, 5),
  (6, 7, 8),
  (0, 3, 6),
  (1, 4, 7),
  (2, 5, 8),
  (0, 4, 8),
  (2, 4, 6),
)

MOVE_ORDER = (4, 0, 2, 6, 8, 1, 3, 5, 7)
POPULATION_SIZE = 512
GENERATIONS = 160
ELITE_FRACTION = 0.15
LEARNING_RATE = 0.85
INIT_SIGMA = 1.0
MIN_SIGMA = 0.05
BAD_MOVE_PENALTY = 20.0
GAP_PENALTY = 5.0


def _build_transforms():
  transforms = []
  for transform in (
    lambda r, c: (r, c),
    lambda r, c: (c, 2 - r),
    lambda r, c: (2 - r, 2 - c),
    lambda r, c: (2 - c, r),
    lambda r, c: (r, 2 - c),
    lambda r, c: (2 - r, c),
    lambda r, c: (c, r),
    lambda r, c: (2 - c, 2 - r),
  ):
    perm = np.empty((BOARD_SIZE,), dtype=np.int8)
    for old_index in range(BOARD_SIZE):
      row, col = divmod(old_index, 3)
      new_row, new_col = transform(row, col)
      perm[new_row * 3 + new_col] = old_index
    transforms.append(perm)
  return tuple(transforms)


TRANSFORMS = _build_transforms()


@dataclass
class TrainingSet:
  boards: np.ndarray
  board_hashes: np.ndarray
  players: np.ndarray
  legal_mask: np.ndarray
  move_values: np.ndarray
  best_values: np.ndarray
  perfect_moves: np.ndarray


def board_to_hash(board):
  board = np.asarray(board, dtype=np.int8).reshape(BOARD_SIZE)
  powers = 3 ** np.arange(BOARD_SIZE, dtype=np.int64)
  return int(np.dot(board.astype(np.int64), powers))


def canonicalize_board(board):
  board = np.asarray(board, dtype=np.int8).reshape(BOARD_SIZE)
  transformed = [(board[perm], perm) for perm in TRANSFORMS]
  canonical, perm = min(transformed, key=lambda item: board_to_hash(item[0]))
  return canonical.copy(), perm


def is_winner(board, player):
  board = np.asarray(board, dtype=np.int8).reshape(BOARD_SIZE)
  return any(np.all(board[list(line)] == player) for line in WIN_LINES)


def is_draw(board):
  board = np.asarray(board, dtype=np.int8).reshape(BOARD_SIZE)
  return not np.any(board == PLAYERS.NONE)


def legal_moves(board):
  board = np.asarray(board, dtype=np.int8).reshape(BOARD_SIZE)
  return np.flatnonzero(board == PLAYERS.NONE)


def current_player_for_board(board):
  board = np.asarray(board, dtype=np.int8).reshape(BOARD_SIZE)
  x_count = int(np.sum(board == PLAYERS.X))
  o_count = int(np.sum(board == PLAYERS.O))
  if x_count == o_count:
    return PLAYERS.X
  if x_count == o_count + 1:
    return PLAYERS.O
  raise ValueError(f'Illegal board counts: X={x_count}, O={o_count}')


def generate_valid_boards():
  boards = []
  players = []
  seen = set()
  board = np.zeros((BOARD_SIZE,), dtype=np.int8)

  def visit(player):
    key = tuple(int(v) for v in board)
    if key in seen:
      return
    seen.add(key)
    if is_winner(board, PLAYERS.X) or is_winner(board, PLAYERS.O) or is_draw(board):
      return
    boards.append(board.copy())
    players.append(player)
    for move in legal_moves(board):
      board[move] = player
      visit(next_player(player))
      board[move] = PLAYERS.NONE

  visit(PLAYERS.X)
  return np.asarray(boards, dtype=np.int8), np.asarray(players, dtype=np.int8)


@lru_cache(maxsize=None)
def _minimax_value(board_tuple, player):
  board = np.asarray(board_tuple, dtype=np.int8)
  opponent = next_player(player)
  if is_winner(board, opponent):
    return -1
  if is_winner(board, player):
    return 1
  if is_draw(board):
    return 0

  best = -2
  for move in legal_moves(board):
    board[move] = player
    if is_winner(board, player):
      value = 1
    elif is_draw(board):
      value = 0
    else:
      value = -_minimax_value(tuple(int(v) for v in board), opponent)
    board[move] = PLAYERS.NONE
    best = max(best, value)
  return best


def move_value(board, player, move):
  board = np.asarray(board, dtype=np.int8).copy().reshape(BOARD_SIZE)
  if board[move] != PLAYERS.NONE:
    return -2
  board[move] = player
  if is_winner(board, player):
    return 1
  if is_draw(board):
    return 0
  return -_minimax_value(tuple(int(v) for v in board), next_player(player))


def perfect_move(board, player=None):
  board = np.asarray(board, dtype=np.int8).reshape(BOARD_SIZE)
  if player is None:
    player = current_player_for_board(board)
  values = {move: move_value(board, player, move) for move in legal_moves(board)}
  best = max(values.values())
  for move in MOVE_ORDER:
    if values.get(move) == best:
      return move
  raise ValueError('No legal move available')


def build_training_set():
  raw_boards, raw_players = generate_valid_boards()
  canonical = {}
  for board, player in zip(raw_boards, raw_players):
    canonical_board, _ = canonicalize_board(board)
    canonical[(board_to_hash(canonical_board), int(player))] = (canonical_board, int(player))
  items = [canonical[key] for key in sorted(canonical)]
  boards = np.asarray([board for board, _ in items], dtype=np.int8)
  players = np.asarray([player for _, player in items], dtype=np.int8)
  board_hashes = np.asarray([board_to_hash(board) for board in boards], dtype=np.int64)
  legal_mask = boards == PLAYERS.NONE
  move_values = np.full((len(boards), BOARD_SIZE), -2.0, dtype=np.float32)
  perfect_moves = np.full((len(boards),), -1, dtype=np.int8)

  for i, (board, player) in enumerate(zip(boards, players)):
    for move in legal_moves(board):
      move_values[i, move] = move_value(board, int(player), int(move))
    best_value = np.max(move_values[i])
    for move in MOVE_ORDER:
      if legal_mask[i, move] and move_values[i, move] == best_value:
        perfect_moves[i] = move
        break

  best_values = np.max(move_values, axis=1)
  return TrainingSet(
    boards=boards,
    board_hashes=board_hashes,
    players=players,
    legal_mask=legal_mask,
    move_values=move_values,
    best_values=best_values.astype(np.float32),
    perfect_moves=perfect_moves,
  )


def evaluate_dna(dna, training_set):
  dna = np.asarray(dna, dtype=np.float32)
  if dna.ndim == 2:
    dna = dna[None, :, :]
  masked_scores = np.where(training_set.legal_mask[None, :, :], dna, -np.inf)
  chosen_moves = np.argmax(masked_scores, axis=2)
  chosen_values = np.take_along_axis(
    training_set.move_values[None, :, :],
    chosen_moves[:, :, None],
    axis=2,
  ).squeeze(axis=2)
  gaps = training_set.best_values[None, :] - chosen_values
  bad_moves = gaps > 0
  fitness = (
    chosen_values.sum(axis=1)
    - BAD_MOVE_PENALTY * bad_moves.sum(axis=1)
    - GAP_PENALTY * gaps.sum(axis=1)
  )
  return fitness, bad_moves.sum(axis=1), chosen_moves


def _params_dna(params):
  if isinstance(params, dict):
    return np.asarray(params['dna'], dtype=np.float32)
  return np.asarray(params, dtype=np.float32)


class Players:
  def __init__(self, params, training_set=None):
    self.training_set = training_set or build_training_set()
    self.params = params if isinstance(params, dict) else {'dna': params}
    self.params['dna'] = _params_dna(self.params)
    self.board_slots = {
      int(board_hash): i for i, board_hash in enumerate(self.training_set.board_hashes)
    }

  @classmethod
  def from_params(cls, params, bs=1, training_set=None):
    dna = _params_dna(params)
    if dna.ndim == 3:
      dna = dna[0]
    dna = np.repeat(dna[None, :, :], bs, axis=0)
    return cls({'dna': dna}, training_set=training_set)

  @property
  def dna(self):
    return self.params['dna']

  @property
  def bs(self):
    return 1 if self.dna.ndim == 2 else self.dna.shape[0]

  def score_board(self, board, current_player=None, player_index=0):
    board = np.asarray(board, dtype=np.int8).reshape(BOARD_SIZE)
    if current_player is None:
      current_player = current_player_for_board(board)
    canonical_board, perm = canonicalize_board(board)
    slot = self.board_slots.get(board_to_hash(canonical_board))
    scores = np.full((BOARD_SIZE,), -np.inf, dtype=np.float32)
    scores[board == PLAYERS.NONE] = 0.0
    if slot is None or self.training_set.players[slot] != current_player:
      return scores
    dna = self.dna if self.dna.ndim == 2 else self.dna[player_index % self.bs]
    canonical_scores = dna[slot].copy()
    scores = np.full((BOARD_SIZE,), -np.inf, dtype=np.float32)
    scores[perm] = canonical_scores
    scores[board != PLAYERS.NONE] = -np.inf
    return scores

  def choose_move(self, board, current_player=None, player_index=0):
    return int(np.argmax(self.score_board(board, current_player, player_index)))

  def play(self, boards, test=False, current_player=PLAYERS.X):
    boards = np.asarray(boards, dtype=np.int8)
    if boards.ndim == 1:
      return self.score_board(boards, current_player)
    return np.asarray([
      self.score_board(board, current_player, i) for i, board in enumerate(boards)
    ], dtype=np.float32)

  def mate(self, rng=None, elite_fraction=ELITE_FRACTION, mutation_sigma=0.35):
    if self.dna.ndim != 3:
      raise ValueError('mate() requires a population DNA tensor')
    rng = rng or np.random.default_rng()
    fitness, _, _ = evaluate_dna(self.dna, self.training_set)
    population_size = self.dna.shape[0]
    elite_count = max(1, int(population_size * elite_fraction))
    elite_indices = np.argsort(fitness)[-elite_count:][::-1]
    elites = self.dna[elite_indices]
    next_dna = np.empty_like(self.dna)
    next_dna[:elite_count] = elites
    parent_indices = rng.integers(0, elite_count, size=population_size - elite_count)
    next_dna[elite_count:] = elites[parent_indices]
    next_dna[elite_count:] += rng.normal(
      0.0,
      mutation_sigma,
      size=next_dna[elite_count:].shape,
    ).astype(np.float32)
    self.params['dna'] = next_dna
    return elite_indices


class Games:
  def __init__(self, bs=1):
    self.bs = bs
    self.boards = np.zeros((bs, BOARD_SIZE), dtype=np.int8)
    self.winners = np.zeros((bs,), dtype=np.int8)
    self.illegal_movers = np.zeros((bs,), dtype=np.int8)
    self.game_over = np.zeros((bs,), dtype=bool)

  def update(self, moves, player, test=False, player_dict=None):
    moves = np.asarray(moves)
    if moves.ndim == 1:
      moves = moves[None, :]
    move_indices = np.argmax(moves, axis=1)
    for i, move in enumerate(move_indices):
      if self.game_over[i]:
        continue
      if self.boards[i, move] != PLAYERS.NONE:
        self.illegal_movers[i] = player
        self.winners[i] = next_player(player)
        self.game_over[i] = True
        continue
      self.boards[i, move] = player
      if is_winner(self.boards[i], player):
        self.winners[i] = player
        self.game_over[i] = True
      elif is_draw(self.boards[i]):
        self.game_over[i] = True

  @property
  def losers(self):
    losers = np.zeros_like(self.winners)
    losers[self.winners == PLAYERS.X] = PLAYERS.O
    losers[self.winners == PLAYERS.O] = PLAYERS.X
    return losers

  @property
  def total_moves(self):
    return float(np.mean(np.sum(self.boards != PLAYERS.NONE, axis=1)))


def init_players(population_size=POPULATION_SIZE, training_set=None, seed=None):
  training_set = training_set or build_training_set()
  rng = np.random.default_rng(seed)
  dna = rng.normal(
    0.0,
    INIT_SIGMA,
    size=(population_size, len(training_set.boards), BOARD_SIZE),
  ).astype(np.float32)
  return Players({'dna': dna}, training_set=training_set)


def play_game(x_player, o_player, x_uses_perfect=False, o_uses_perfect=False):
  board = np.zeros((BOARD_SIZE,), dtype=np.int8)
  current_player = PLAYERS.X
  while True:
    if current_player == PLAYERS.X:
      move = perfect_move(board, PLAYERS.X) if x_uses_perfect else x_player.choose_move(board, PLAYERS.X)
    else:
      move = perfect_move(board, PLAYERS.O) if o_uses_perfect else o_player.choose_move(board, PLAYERS.O)
    if board[move] != PLAYERS.NONE:
      return {
        'winner': next_player(current_player),
        'moves': int(np.sum(board != PLAYERS.NONE)),
        'illegal_move': current_player,
      }
    board[move] = current_player
    if is_winner(board, current_player):
      return {
        'winner': current_player,
        'moves': int(np.sum(board != PLAYERS.NONE)),
        'illegal_move': PLAYERS.NONE,
      }
    if is_draw(board):
      return {
        'winner': PLAYERS.NONE,
        'moves': int(np.sum(board != PLAYERS.NONE)),
        'illegal_move': PLAYERS.NONE,
      }
    current_player = next_player(current_player)


def count_losing_moves(player_instance):
  raw_boards, raw_players = generate_valid_boards()
  bad_moves = 0
  for board, player in zip(raw_boards, raw_players):
    values = [move_value(board, int(player), int(move)) for move in legal_moves(board)]
    best_value = max(values)
    chosen_move = player_instance.choose_move(board, int(player))
    if move_value(board, int(player), chosen_move) < best_value:
      bad_moves += 1
  return bad_moves, len(raw_boards)


def get_losing_move_ratio(player_instance):
  bad_moves, total = count_losing_moves(player_instance)
  return float(bad_moves / total)


def evaluate_against_perfect(player_instance):
  player = Players.from_params(player_instance.params, training_set=player_instance.training_set)
  as_x = play_game(player, player, o_uses_perfect=True)
  as_o = play_game(player, player, x_uses_perfect=True)
  return {
    'x_moves': as_x['moves'],
    'x_draw_rate': 1.0 if as_x['winner'] == PLAYERS.NONE else 0.0,
    'x_loss_rate': 1.0 if as_x['winner'] == PLAYERS.O else 0.0,
    'o_moves': as_o['moves'],
    'o_draw_rate': 1.0 if as_o['winner'] == PLAYERS.NONE else 0.0,
    'o_loss_rate': 1.0 if as_o['winner'] == PLAYERS.X else 0.0,
    'illegal_moves': int(as_x['illegal_move'] != PLAYERS.NONE) + int(as_o['illegal_move'] != PLAYERS.NONE),
  }


def train_run(
  name='',
  population_size=POPULATION_SIZE,
  generations=GENERATIONS,
  seed=0,
  save_path='organic_dna.pkl',
  progress=True,
):
  training_set = build_training_set()
  rng = np.random.default_rng(seed)
  mean = np.zeros((len(training_set.boards), BOARD_SIZE), dtype=np.float32)
  sigma = np.full_like(mean, INIT_SIGMA, dtype=np.float32)
  best_dna = None
  best_fitness = -np.inf
  best_bad_count = len(training_set.boards)
  elite_count = max(1, int(population_size * ELITE_FRACTION))
  history = []

  for generation in range(generations):
    population = rng.normal(
      mean,
      sigma,
      size=(population_size, len(training_set.boards), BOARD_SIZE),
    ).astype(np.float32)
    if best_dna is not None:
      population[0] = best_dna

    fitness, bad_counts, _ = evaluate_dna(population, training_set)
    best_index = int(np.argmax(fitness))
    if fitness[best_index] > best_fitness:
      best_fitness = float(fitness[best_index])
      best_bad_count = int(bad_counts[best_index])
      best_dna = population[best_index].copy()

    elite_indices = np.argsort(fitness)[-elite_count:]
    elites = population[elite_indices]
    elite_mean = elites.mean(axis=0)
    elite_sigma = np.maximum(elites.std(axis=0), MIN_SIGMA)
    mean = ((1.0 - LEARNING_RATE) * mean + LEARNING_RATE * elite_mean).astype(np.float32)
    sigma = ((1.0 - LEARNING_RATE) * sigma + LEARNING_RATE * elite_sigma).astype(np.float32)

    history.append({
      'generation': generation,
      'best_bad_moves': best_bad_count,
      'best_fitness': best_fitness,
      'mean_bad_moves': float(np.mean(bad_counts)),
    })
    if progress and (generation % 10 == 0 or best_bad_count == 0):
      print(
        f'generation={generation} '
        f'best_bad_moves={best_bad_count} '
        f'mean_bad_moves={np.mean(bad_counts):.2f}'
      )
    if best_bad_count == 0:
      break

  params = {
    'dna': best_dna,
    'board_hashes': training_set.board_hashes,
    'history': history,
    'seed': seed,
    'population_size': population_size,
    'generations': generations,
  }
  player = Players(params, training_set=training_set)
  if save_path:
    with open(save_path, 'wb') as f:
      pickle.dump(params, f)
  return player


def load_players(path='organic_dna.pkl'):
  with open(path, 'rb') as f:
    params = pickle.load(f)
  return Players(params)


def generate_perfect_moves():
  training_set = build_training_set()
  return {
    int(board_hash): int(move)
    for board_hash, move in zip(training_set.board_hashes, training_set.perfect_moves)
  }


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--population-size', '--bs', type=int, default=POPULATION_SIZE)
  parser.add_argument('--generations', '--steps', type=int, default=GENERATIONS)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--runs', type=int, default=1)
  parser.add_argument('--save-path', default='organic_dna.pkl')
  parser.add_argument('--no-save', action='store_true')
  parser.add_argument('--no-progress', action='store_true')
  return parser.parse_args()


if __name__ == '__main__':
  args = parse_args()
  for run in range(args.runs):
    save_path = None if args.no_save else args.save_path
    if save_path and args.runs > 1:
      path = Path(args.save_path)
      suffix = path.suffix or '.pkl'
      save_path = str(path.with_name(f'{path.stem}_{run}{suffix}'))
    player = train_run(
      population_size=args.population_size,
      generations=args.generations,
      seed=args.seed + run,
      save_path=save_path,
      progress=not args.no_progress,
    )
    print(evaluate_against_perfect(player))
