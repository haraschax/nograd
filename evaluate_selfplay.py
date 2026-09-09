#!/usr/bin/env python3
"""Audit a frozen genome independently; never imported by the trainer.

This is the only new module containing a perfect-play solver. Its bitboard
rules and state enumeration are independent of the trainer's tuple rules.
"""

import argparse
from functools import lru_cache
import json
from selfplay import load_checkpoint


FULL = (1 << 9) - 1
WIN_MASKS = (0b000000111, 0b000111000, 0b111000000,
             0b001001001, 0b010010010, 0b100100100,
             0b100010001, 0b001010100)


def terminal(x, o):
    for player, bits in ((1, x), (2, o)):
        if any(bits & mask == mask for mask in WIN_MASKS):
            return player
    return 0 if x | o == FULL else None


def next_state(x, o, move):
    bit = 1 << move
    if x.bit_count() == o.bit_count():
        return x | bit, o
    return x, o | bit


def moves(x, o):
    return [i for i in range(9) if not (x | o) & (1 << i)]


@lru_cache(maxsize=None)
def value(x, o):
    """Exact value for the player whose turn follows this position."""
    result = terminal(x, o)
    if result is not None:
        player = 1 if x.bit_count() == o.bit_count() else 2
        return 0 if result == 0 else (1 if result == player else -1)
    return max(-value(*next_state(x, o, move)) for move in moves(x, o))


def as_board(x, o):
    return tuple(1 if x & (1 << i) else 2 if o & (1 << i) else 0
                 for i in range(9))


def audit(genome):
    seen = set()
    stats = {"positions": 0, "illegal_moves": 0, "suboptimal_moves": 0}

    def visit(x, o):
        if (x, o) in seen:
            return
        seen.add((x, o))
        if terminal(x, o) is not None:
            return
        stats["positions"] += 1
        try:
            move = genome.move(as_board(x, o))
        except ValueError:
            move = -1
        if move not in moves(x, o):
            stats["illegal_moves"] += 1
        elif -value(*next_state(x, o, move)) != value(x, o):
            stats["suboptimal_moves"] += 1
        for candidate in moves(x, o):
            visit(*next_state(x, o, candidate))

    visit(0, 0)
    for role in (1, 2):
        x = o = 0
        while terminal(x, o) is None:
            player = 1 if x.bit_count() == o.bit_count() else 2
            if player == role:
                move = genome.move(as_board(x, o))
            else:
                move = max(moves(x, o), key=lambda m: -value(*next_state(x, o, m)))
            x, o = next_state(x, o, move)
        stats["as_x" if role == 1 else "as_o"] = {
            "winner": terminal(x, o), "moves": (x | o).bit_count()}
    return stats


def passed(stats):
    return (stats["positions"] == 4520 and stats["illegal_moves"] == 0
            and stats["suboptimal_moves"] == 0
            and all(stats[role] == {"winner": 0, "moves": 9}
                    for role in ("as_x", "as_o")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", default="organic_dna.json")
    args = parser.parse_args()
    stats = audit(load_checkpoint(args.checkpoint))
    print(json.dumps(stats, sort_keys=True))
    raise SystemExit(0 if passed(stats) else 1)


if __name__ == "__main__":
    main()
