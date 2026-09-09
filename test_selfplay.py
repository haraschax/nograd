"""Regression checks for learned correctness and isolation from the oracle."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from selfplay import (EMPTY, after_move, board_key, decision_boards,
                      load_checkpoint, save_checkpoint, train)


ROOT = Path(__file__).resolve().parent


class SelfPlayTests(unittest.TestCase):
    def test_converges_from_ten_random_seeds(self):
        # Import the oracle only in a separate audit process after each frozen
        # checkpoint has been saved. Its result is never fed back to train().
        with tempfile.TemporaryDirectory() as directory:
            for seed in range(10):
                with self.subTest(seed=seed):
                    genome, stats = train(seed)
                    self.assertEqual(stats["positions"], 4520)
                    self.assertEqual(stats["games"], 16167)
                    self.assertGreater(stats["accepted_mutations"], 0)
                    path = Path(directory) / f"seed{seed}.json"
                    save_checkpoint(path, genome, stats)
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    result = subprocess.run(
                        [sys.executable, str(ROOT / "evaluate_selfplay.py"), str(path)],
                        check=True, capture_output=True, text=True)
                    audit = json.loads(result.stdout)
                    self.assertEqual(audit["suboptimal_moves"], 0)
                    self.assertEqual(audit["illegal_moves"], 0)
                    self.assertEqual(audit["as_x"], {"winner": 0, "moves": 9})
                    self.assertEqual(audit["as_o"], {"winner": 0, "moves": 9})
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_training_is_identical_without_evaluator_or_perfect_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            shutil.copyfile(ROOT / "selfplay.py", directory / "selfplay.py")
            subprocess.run([sys.executable, "-I", str(directory / "selfplay.py"),
                            "--seed", "23", "--output", str(directory / "isolated.json")],
                           cwd=directory, check=True, capture_output=True, text=True)
            genome, stats = train(23)
            save_checkpoint(directory / "ordinary.json", genome, stats)
            self.assertEqual((directory / "isolated.json").read_bytes(),
                             (directory / "ordinary.json").read_bytes())

    def test_evaluator_detects_a_legal_but_losing_mutation(self):
        genome, stats = train(0)
        board = (1, 1, 0, 2, 2, 0, 0, 0, 0)
        genome.dna[board_key(board)] = 8  # Ignores own win and opponent's threat.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            save_checkpoint(path, genome, stats)
            result = subprocess.run(
                [sys.executable, str(ROOT / "evaluate_selfplay.py"), str(path)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertGreater(json.loads(result.stdout)["suboptimal_moves"], 0)

    def test_openings_stop_at_a_win(self):
        boards = set(decision_boards())
        self.assertEqual(len(boards), 4520)
        board = EMPTY
        for move in (0, 3, 1, 4, 2):
            board = after_move(board, move)
        self.assertNotIn(board, boards)
        self.assertNotIn(after_move(board, 8), boards)

    def test_checkpoint_roundtrip_and_missing_allele(self):
        genome, stats = train(5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dna.json"
            save_checkpoint(path, genome, stats)
            self.assertEqual(load_checkpoint(path).dna, genome.dna)
            payload = json.loads(path.read_text())
            del payload["genes"]["0"]
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_checkpoint(path)

    def test_play_cli(self):
        genome, stats = train(2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dna.json"
            save_checkpoint(path, genome, stats)
            result = subprocess.run(
                [sys.executable, str(ROOT / "play_selfplay.py"), "--self-play",
                 "--checkpoint", str(path)], check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout), {"winner": 0, "moves": 9})


if __name__ == "__main__":
    unittest.main()
