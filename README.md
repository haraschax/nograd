# Learning to play perfect tic-tac-toe without gradient descent
Most of the world's learning is just DNA-replication + mutation. Less than 1% of the biomass of all life is of organisms that have any neurons at all. That means over 99% of all life learns with DNA-replication + mutation alone. However, no modern ML techniques look anything like this. I think that should change. This repo can produce a perfect tic-tac-toe player in less than 200 lines of code using DNA-like learning. There is no optimizer, no gradients, and no loss function. It is more robust, conceptually simpler, and I think far more beautiful than normal ML techniques to solve tic-tac-toe.

# Some findings during this project
- ML techniques are far more robust (if the loss function is good) than classical algorithms, that is a great advantage of ML, but also it's biggest pitfall. ML has a tendency to hide small bugs by having good performance despite them. This DNA-based learning has the same characteristics tenfold. These agents learn to survive in even the most broken simulation enviroments.
- The beauty of this type of learning is that hyperparameters are trivial to add to the learning process. Mutation rate of the DNA has a massive effect on how quickly it learns. And it should be different at the beginning of training than end. No problem, just add it to the DNA and it will figure itself out.
