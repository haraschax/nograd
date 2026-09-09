#!/usr/bin/env python3
"""Evolve a table genome using terminal outcomes of curriculum self-play.

Training uses legal openings and terminal game outcomes, without externally
solved positions, opponent oracles, gradients, or pre-trained players. The
exhaustive backward curriculum is a form of retrograde policy improvement;
see README.md for the change from a neural to a position-indexed genome.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random


EMPTY = (0,) * 9
LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
         (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))
POWERS = tuple(3 ** i for i in range(9))
UNSET = 255


def board_key(board):
    return sum(cell * power for cell, power in zip(board, POWERS))


def winner(board):
    for a, b, c in LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return 0


def current_player(board):
    return 1 if board.count(1) == board.count(2) else 2


def legal_moves(board):
    return [i for i, cell in enumerate(board) if cell == 0]


def after_move(board, move):
    if move not in range(9) or board[move] != 0:
        raise ValueError("Genome selected an illegal move")
    return board[:move] + (current_player(board),) + board[move + 1:]


def decision_boards():
    """Enumerate legal openings, stopping immediately when somebody wins.

    This enumerates positions only: it never computes game values or moves.
    """
    seen = set()
    boards = []

    def visit(board):
        if board in seen:
            return
        seen.add(board)
        if winner(board) or 0 not in board:
            return
        boards.append(board)
        for move in legal_moves(board):
            visit(after_move(board, move))

    visit(EMPTY)
    return boards


@dataclass
class Genome:
    dna: bytearray

    def move(self, board):
        move = self.dna[board_key(board)]
        if move not in range(9) or board[move]:
            raise ValueError("Genome has no legal allele for this position")
        return move

    def offspring(self, board, move):
        child = Genome(self.dna.copy())
        child.dna[board_key(board)] = move
        return child


def play_game(x_genome, o_genome, opening=EMPTY):
    """Return terminal winner and final board from one actual game."""
    board = opening
    while not winner(board) and 0 in board:
        player = x_genome if current_player(board) == 1 else o_genome
        board = after_move(board, player.move(board))
    return winner(board), board


def outcome(genome, opponent, opening):
    """Win/draw/loss for the genome taking the next turn in this opening."""
    player = current_player(opening)
    x, o = (genome, opponent) if player == 1 else (opponent, genome)
    result, _ = play_game(x, o, opening)
    return 0 if result == 0 else (1 if result == player else -1)


def train(seed=0, progress=None):
    rng = random.Random(seed)
    boards = decision_boards()
    genome = Genome(bytearray([UNSET]) * (3 ** 9))
    for board in boards:
        genome.dna[board_key(board)] = rng.choice(legal_moves(board))

    stats = {"seed": seed, "positions": len(boards), "games": 0,
             "offspring": 0, "accepted_mutations": 0, "stages": []}

    # Later decisions are learned first. Within one stage, no game's future
    # can encounter another position in that stage, since every move fills
    # one square. Their order therefore supplies no privileged strategy.
    for occupied in range(8, -1, -1):
        stage = [board for board in boards if 9 - board.count(0) == occupied]
        rng.shuffle(stage)
        accepted_before = stats["accepted_mutations"]
        for board in stage:
            # Keep the opponent fixed throughout this family tournament.
            opponent = genome
            parent_score = outcome(genome, opponent, board)
            stats["games"] += 1
            mutations = [move for move in legal_moves(board)
                         if move != genome.move(board)]
            rng.shuffle(mutations)
            for move in mutations:
                child = genome.offspring(board, move)
                child_score = outcome(child, opponent, board)
                stats["games"] += 1
                stats["offspring"] += 1
                if child_score > parent_score:
                    genome, parent_score = child, child_score
                    stats["accepted_mutations"] += 1
        record = {"occupied": occupied, "positions": len(stage),
                  "accepted_mutations": stats["accepted_mutations"] - accepted_before}
        stats["stages"].append(record)
        if progress is not None:
            progress(record)
    return genome, stats


def save_checkpoint(path, genome, stats):
    payload = {"format": "nograd-table-dna-v1", "training": stats,
               "genes": {str(key): move for key, move in enumerate(genome.dna)
                         if move != UNSET}}
    Path(path).write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def load_checkpoint(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "nograd-table-dna-v1":
        raise ValueError("Unsupported checkpoint format")
    dna = bytearray([UNSET]) * (3 ** 9)
    for key, move in payload["genes"].items():
        index = int(key)
        if not 0 <= index < len(dna) or type(move) is not int or not 0 <= move < 9:
            raise ValueError("Invalid checkpoint allele")
        dna[index] = move
    genome = Genome(dna)
    for board in decision_boards():
        genome.move(board)
    return genome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="organic_dna.json")
    args = parser.parse_args()
    genome, stats = train(args.seed)
    save_checkpoint(args.output, genome, stats)
    print(json.dumps(stats, sort_keys=True))
    print(f"Saved evolved genome to {args.output}")


if __name__ == "__main__":
    main()
