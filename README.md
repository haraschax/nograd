# Learning to play perfect tic-tac-toe without gradient descent
Less than 1% of the biomass of all life is of organisms that have any neurons at all. That means over 99% of all life learns with DNA-replication + mutation alone. However, no modern ML techniques look anything like this. That should change. This repo can produce a perfect tic-tac-toe player in ~300 lines of code using DNA-like learning. There is no optimizer, no gradients, and no loss function. It is more robust, conceptually simpler, and far more beautiful than conventional ML techniques to solve tic-tac-toe.

## Results
A few dozen training runs look like this:
![Screenshot from 2024-06-27 20-27-58](https://github.com/haraschax/nograd/assets/6804392/289e2d43-4dca-4dc2-be43-d1a7c1fd97c2)
When playing perfectly against a perfect player, games should last 9 moves. As you can see the DNA-based players learn to play 9 moves quite quickly. And training tends to converge at similar speed.



## How to use

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
