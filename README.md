# Learning to play perfect tic-tac-toe without gradient descent

Less than 1% of all life has neurons. Most life still adapts through DNA copy and mutation. This repo uses that idea to train a tic-tac-toe player. There is no optimizer, no gradient step, and no loss function.

## Results

The original project used plots like this to check whether games against a perfect player reached 9 moves.

![Screenshot from 2024-06-27 20-27-58](https://github.com/haraschax/nograd/assets/6804392/289e2d43-4dca-4dc2-be43-d1a7c1fd97c2)

Training now evolves a small DNA table. It does not seed part of the population with a perfect player. Each genome stores move scores for each board state. Rotation and mirror states share the same genes.

Selection rewards genomes that keep the board value safe. Mutation then explores nearby DNA tables. The trained player validates as perfect when it draws against a minimax-perfect opponent as both X and O. A perfect draw lasts 9 moves.

## How to use

To train a tic-tac-toe player:

```
./batch_arena.py
```

To run several convergence checks:

```
./batch_arena.py --runs 5 --population-size 512 --generations 160 --no-progress --no-save
```

To play against one of the players from the population you just trained:

```
./play.py
```

![Screenshot from 2024-06-27 20-47-01](https://github.com/haraschax/nograd/assets/6804392/cdedd0d5-75d1-4a63-bb73-b464409dcde0)

To play against a minimax-perfect player:

```
./play.py --perfect
```

## Fixed hacks

- The population no longer contains hardcoded perfect players.
- Training no longer depends on committed `perfect_dna.pkl` or `perfect_moves.pkl` artifacts.
- Training and validation run on CPU without a CUDA-only startup assumption.
