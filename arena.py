


from agent import Agent, DNA_SIZE
import torch
from random import sample
#from tictactoe import TicTacToe as Game
from dummy_game import DummyGame as Game




AGENTS_N = 100



agents = {}
for i in range(AGENTS_N):
  random_dna = torch.randn(DNA_SIZE)
  agent = Agent(random_dna)
  agents[agent.hash] = agent



game_cnt = 0
total_moves = []

while True:
  player1_hash, player2_hash = sample(list(agents.keys()), 2)
  player1, player2 = agents[player1_hash], agents[player2_hash]
  game = Game()
  while not game.isover:
    if game.last_player == 1 or game.last_player == 0:
      action = player1.act(game.board)
      game.move(*action, -1)
    else:
      action = player2.act(game.board)
      game.move(*action, 1)
  if game.winner == -1:
    player1.credits += 1
    player2.credits -= 1
    #print(f" Player 1 wins after {game.total_moves} moves , credits = {player1.credits}")
  elif game.winner == 1:
    player1.credits -= 1
    player2.credits += 1
    #print(f" Player 2 wins after {game.total_moves} moves , credits = {player2.credits}")
  else:
    pass#print(f" Nobody wins after {game.total_moves} moves")
  game_cnt += 1
  total_moves.append(game.total_moves)
  if game_cnt % 10000 == 0:
    print(f"Average moves per game: {sum(total_moves) / len(total_moves)}")
    total_moves = []


  for agent in list(agents.values()):
    if agent.credits <= 0:
      agents.pop(agent.hash)
    if agent.credits >= 10:
      new_dna = agent.give_birth()
      #new_dna = torch.randn(DNA_SIZE)
      new_agent = Agent(new_dna)
      agents[new_agent.hash] = new_agent



  