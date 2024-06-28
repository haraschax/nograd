# Learning to play perfect tic-tac-toe without gradient descent
Less than 1% of the biomass of all life is of organisms that have any neurons at all. That means over 99% of all life learns with DNA-replication + mutation alone. However, no modern ML techniques look anything like this. That should change. This repo can produce a perfect tic-tac-toe player in ~300 lines of code using DNA-like learning. There is no optimizer, no gradients, and no loss function. It is more robust, conceptually simpler, and far more beautiful than conventional ML techniques to solve tic-tac-toe.

## How to use

To train a tic-tac-toe player just run:
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
