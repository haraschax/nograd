# Learning to play perfect tic-tac-toe without gradient descent
Less than 1% of the biomass of all life is of organisms that have any neurons at all. That means over 99% of all life learns with DNA-replication + mutation alone. However, no modern ML techniques look anything like this. That should change. This repo can produce a perfect tic-tac-toe player in ~300 lines of code using DNA-like learning. There is no optimizer, no gradients, and no loss function. It is more robust, conceptually simpler, and far more beautiful than conventional ML techniques to solve tic-tac-toe.

## Results
A few dozen training runs look like this:
![Screenshot from 2024-06-27 20-27-58](https://github.com/haraschax/nograd/assets/6804392/289e2d43-4dca-4dc2-be43-d1a7c1fd97c2)
When playing perfectly against a perfect player, games should last 9 moves. As you can see the DNA-based players learn to play 9 moves quite quickly. And training tends to converge at similar speed.



## How to use

### Oracle-free table-genome training

The new `selfplay.py` provides a reproducible CPU-only path that starts with
random legal moves and learns from terminal self-play outcomes. It requires
Python 3.10 or newer and only the standard library:

```sh
python selfplay.py --seed 0 --output organic_dna.json
python evaluate_selfplay.py organic_dna.json
python play_selfplay.py --checkpoint organic_dna.json
# To play as O, add --human o; for an automatic game, add --self-play.
python -m unittest -v test_selfplay.py
```

The checkpoint is a JSON genome; no saved perfect player or move dataset is
read. Training does not import the evaluator. `evaluate_selfplay.py` uses an
independent bitboard implementation and a perfect-play solver **only after
training**, to audit the frozen checkpoint. It exits unsuccessfully for any
illegal or suboptimal move.

#### What changed, and what did not get solved

This is an alternative representation and training algorithm, rather than a
convergence fix for the original neural population. Each gene contains one
legal move for one reachable board. All 4,520 nonterminal reachable boards
are used as openings. Boards are learned from eight occupied squares down
to the empty board; the order within each stage is shuffled.

At each opening, the current genome plays a complete game against a frozen
copy of itself. Offspring replicate its DNA and mutate that opening's move.
Every other legal allele is tried once, in random order. Each offspring
plays a complete game against the same opponent, with win/draw/loss scored
as 1/0/-1 for the player taking the next turn. An offspring replaces its
parent only when its terminal outcome improves. Later-position genes remain
unchanged during that family tournament. There are no tactical rules,
pre-trained players, solver-derived labels, gradients, or intermediate
position values in this training path.

The exhaustive backward curriculum is essential to reliability. Once all
later decisions have been learned, complete games correctly distinguish the
current opening's winning, drawing, and losing mutations. Backward induction
therefore establishes optimal play at every opening. Mathematically this is
tabular retrograde policy improvement implemented through mutation and
self-play games; it does not demonstrate generalization or solve the
scalability problem for larger games. It does not establish convergence of
the original neural architecture.

Validation runs seeds 0 through 9 independently. Each run uses 16,167 games
and 11,647 offspring, then achieves zero illegal and zero suboptimal moves
over all 4,520 nonterminal legal positions, and nine-move draws against a
perfect player as both X and O. The tests also train in an isolated directory
containing only `selfplay.py`, verify that its checkpoint is byte-identical
to ordinary training, and check that evaluation leaves the checkpoint
unchanged. Thus the test-time oracle cannot guide selection or stopping.

### Original neural-genome experiment

The original CUDA training and Pygame scripts remain available below. They
have their original dependencies and convergence limitations. The new
trainer does not import or modify their saved genomes.

To train a tic-tac-toe player just run (can be visualized with tensorboard):
```
./batch_arena.py
```

To play against one of the players from the population you just trained:
```
./play.py
```
![Screenshot from 2024-06-27 20-47-01](https://github.com/haraschax/nograd/assets/6804392/cdedd0d5-75d1-4a63-bb73-b464409dcde0)

To play against a perfect player that was trained classically:
```
./play.py --perfect
```


## Hacks still to fix
- 20% of the population is hardcoded to be a perfect player, without that convergence is not reliable
- The architecture is handcoded and arbitrary, that should ideally also be learned
