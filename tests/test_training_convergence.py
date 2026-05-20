import unittest

import numpy as np

from helpers import BOARD_SIZE, PLAYERS
from batch_arena import (
  Players,
  build_training_set,
  evaluate_against_perfect,
  evaluate_dna,
  count_losing_moves,
  init_players,
  perfect_move,
  train_run,
)


class TrainingRulesTest(unittest.TestCase):
  def test_training_set_contains_only_decision_states(self):
    training_set = build_training_set()

    self.assertGreater(len(training_set.boards), 0)
    self.assertEqual(training_set.boards.shape[1], BOARD_SIZE)
    self.assertTrue(np.all(training_set.legal_mask.any(axis=1)))
    self.assertTrue(np.all(training_set.perfect_moves >= 0))

  def test_perfect_move_takes_immediate_win(self):
    board = np.array([
      PLAYERS.X, PLAYERS.X, PLAYERS.NONE,
      PLAYERS.O, PLAYERS.O, PLAYERS.NONE,
      PLAYERS.NONE, PLAYERS.NONE, PLAYERS.NONE,
    ], dtype=np.int8)

    self.assertEqual(perfect_move(board, PLAYERS.X), 2)

  def test_perfect_move_blocks_immediate_loss(self):
    board = np.array([
      PLAYERS.X, PLAYERS.X, PLAYERS.NONE,
      PLAYERS.NONE, PLAYERS.O, PLAYERS.NONE,
      PLAYERS.NONE, PLAYERS.NONE, PLAYERS.NONE,
    ], dtype=np.int8)

    self.assertEqual(perfect_move(board, PLAYERS.O), 2)

  def test_mate_keeps_population_size_and_elites(self):
    training_set = build_training_set()
    players = init_players(population_size=16, training_set=training_set, seed=7)
    before = players.dna.copy()

    elite_indices = players.mate(rng=np.random.default_rng(8), elite_fraction=0.25)

    self.assertEqual(players.dna.shape, before.shape)
    self.assertEqual(len(elite_indices), 4)
    np.testing.assert_allclose(players.dna[0], before[elite_indices[0]])

  def test_short_training_converges_to_no_losing_moves(self):
    player = train_run(
      population_size=512,
      generations=160,
      seed=0,
      save_path=None,
      progress=False,
    )
    _, bad_counts, _ = evaluate_dna(player.dna, player.training_set)
    self.assertEqual(int(bad_counts[0]), 0)
    self.assertEqual(count_losing_moves(player), (0, 4520))

  def test_trained_player_draws_perfect_as_x_and_o(self):
    player = train_run(
      population_size=512,
      generations=160,
      seed=1,
      save_path=None,
      progress=False,
    )
    result = evaluate_against_perfect(Players.from_params(player.params, training_set=player.training_set))

    self.assertEqual(result['x_moves'], 9)
    self.assertEqual(result['x_loss_rate'], 0.0)
    self.assertEqual(result['o_moves'], 9)
    self.assertEqual(result['o_loss_rate'], 0.0)
    self.assertEqual(result['illegal_moves'], 0)


if __name__ == '__main__':
  unittest.main()
