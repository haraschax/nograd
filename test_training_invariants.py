import unittest

import batch_arena


class TrainingInvariantTests(unittest.TestCase):
  def test_mate_keeps_population_tensors_aligned(self):
    batch_arena.DEVICE = 'cpu'
    players = batch_arena.init_players(bs=4)
    before = players.params['dna'].shape[0]

    players.mate()

    self.assertEqual(players.params['dna'].shape[0], before)
    for name, value in players.params.items():
      self.assertEqual(value.shape[0], before, name)


if __name__ == '__main__':
  unittest.main()
