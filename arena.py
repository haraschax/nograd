


from agent import Agent, DNA_SIZE, INIT_CREDS
import torch
torch.set_grad_enabled(False)
from random import sample
from tictactoe import TicTacToe as Game
#from dummy_game import DummyGame as Game








class Arena():
  def __init__(self, agents):
    self.game_cnt = 0
    self.total_moves = []
    self.births = 0
    self.last_mute = 0
    self.last_mute2 = 0
    self.agents = agents

  def one_round(self):
    player1_hash, player2_hash = sample(list(self.agents.keys()), 2)
    player1, player2 = self.agents[player1_hash], self.agents[player2_hash]
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
    self.game_cnt += 1
    self.total_moves.append(game.total_moves)


    for agent in list(self.agents.values()):
      if agent.credits <= 0:
        self.agents.pop(agent.hash)
      if agent.credits >= INIT_CREDS * 2:
        self.births += 1
        new_dna = agent.give_birth()
        self.last_mute = agent.mutation_rate
        self.last_mute2 = agent.mutation_size
        new_agent = Agent(new_dna)
        self.agents[new_agent.hash] = new_agent

        if agent.births >= 20:
          self.births += 1
          new_dna = agent.give_birth()
          new_agent = Agent(new_dna)
          self.agents[new_agent.hash] = new_agent
          self.agents.pop(agent.hash)



  def k_rounds(self, k):
    for _ in range(k):
      self.one_round()


AGENTS_N = 10000

agents = {}
for i in range(AGENTS_N):
  agent = Agent()
  agents[agent.hash] = agent


arena = Arena(agents)


import cProfile
cProfile.run('arena.k_rounds(1000)', 'restats')

import pstats
from pstats import SortKey
p = pstats.Stats('restats')
p.sort_stats(SortKey.CUMULATIVE).print_stats(50)

while True:
  arena.k_rounds(1000)
  print(f"Average moves per game: {sum(arena.total_moves) / len(arena.total_moves)}, full games: {sum(torch.tensor(arena.total_moves) == 9) / len(arena.total_moves):.2g}, total agents: {len(arena.agents)}")
  print(f"total games: {arena.game_cnt} total births: {arena.births}, mut rate {arena.last_mute:.2g}, mut size {arena.last_mute2:.2g}")
  arena.total_moves = []
