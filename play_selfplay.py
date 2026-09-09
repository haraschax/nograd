#!/usr/bin/env python3
"""Play an evolved table genome in a terminal, with no optional dependencies."""

import argparse
import json
from selfplay import (EMPTY, after_move, current_player, legal_moves,
                      load_checkpoint, play_game, winner)


def display(board):
    cells = ["X" if cell == 1 else "O" if cell == 2 else str(i + 1)
             for i, cell in enumerate(board)]
    print("\n" + "\n---+---+---\n".join(
        " " + " | ".join(cells[start:start + 3]) for start in (0, 3, 6)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="organic_dna.json")
    parser.add_argument("--human", choices=("x", "o"), default="x")
    parser.add_argument("--self-play", action="store_true",
                        help="Play both sides automatically and print the result")
    args = parser.parse_args()
    genome = load_checkpoint(args.checkpoint)
    if args.self_play:
        result, board = play_game(genome, genome)
        print(json.dumps({"winner": result, "moves": 9 - board.count(0)}))
        return

    human = 1 if args.human == "x" else 2
    board = EMPTY
    while not winner(board) and 0 in board:
        display(board)
        if current_player(board) == human:
            try:
                move = int(input("Your move (1-9): ")) - 1
            except ValueError:
                print("Choose a number from 1 to 9.")
                continue
            except (EOFError, KeyboardInterrupt):
                print("\nGame ended.")
                return
            if move not in legal_moves(board):
                print("Choose an empty square.")
                continue
        else:
            move = genome.move(board)
            print(f"Genome plays {move + 1}.")
        board = after_move(board, move)
    display(board)
    result = winner(board)
    print("Draw." if result == 0 else ("X wins." if result == 1 else "O wins."))


if __name__ == "__main__":
    main()
