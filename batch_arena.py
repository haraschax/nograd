
MAX_MOVES = 10
BATCH_SIZE = 1000
BOARD_SIZE = 9
INIT_CREDS = 5

import torch
import time

class PLAYERS:
  NONE = 0
  X = 1
  O = 2


class Games():
  def __init__(self):
    self.boards = torch.zeros((BATCH_SIZE, BOARD_SIZE), dtype=torch.int8)
    self.winners = torch.zeros((BATCH_SIZE,), dtype=torch.int8)
    self.update_game_over()

  def update(self, moves, player):
    assert len(moves) == BATCH_SIZE
    assert len(moves) == self.boards.shape[0]
    move_idxs = torch.argmax(moves, dim=1, keepdim=True)
  
    illegal_movers = self.boards.gather(1, move_idxs) != PLAYERS.NONE
    self.winners[illegal_movers[:,0]] = PLAYERS.O if player == PLAYERS.X else PLAYERS.X

    move_scattered = torch.zeros_like(self.boards.to(dtype=torch.bool))
    move_scattered.scatter_(1, move_idxs, 1)

    self.boards = self.boards + move_scattered * player
 

   
    self.update_game_over()
  
  def check_winners(self):
    for idx, board_flat in enumerate(self.boards):
      board = board_flat.reshape((3,3))
      for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != 0:
          self.winners[idx] = board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != 0:
          self.winners[idx] = board[0][i]
      if board[0][0] == board[1][1] == board[2][2] != 0:
        self.winners[idx] = board[0][0]

  @property
  def losers(self):
    losers = torch.zeros_like(self.winners)
    losers[self.winners == PLAYERS.X] = PLAYERS.O
    losers[self.winners == PLAYERS.O] = PLAYERS.X
    return losers
  
  @property
  def total_moves(self):
    return (self.boards != PLAYERS.NONE).sum(dim=1).float().mean()

  def update_game_over(self):
    self.game_over = (self.winners != PLAYERS.NONE) | ((self.boards != PLAYERS.NONE).sum(dim=1) == BOARD_SIZE)      

class Players():
  def __init__(self, credits, action_matrix):
    self.credits = credits
    self.action_matrix = action_matrix

  def play(self, boards):
    moves = torch.einsum('bij, bj->bi', self.action_matrix, boards.float())
    return moves
  
  def mate(self):
    mutation_rate = 1e-3
    new_action_matrix = self.action_matrix + torch.randn(self.action_matrix.shape) * (torch.rand(self.action_matrix.shape) < mutation_rate)
    dead = (self.credits == 0).nonzero(as_tuple=True)[0]
    can_mate = (self.credits > 5).nonzero(as_tuple=True)[0]
    assert len(can_mate) >= len(dead)
    self.credits[dead] = INIT_CREDS
    self.credits[can_mate[:len(dead)]] -= INIT_CREDS
    self.action_matrix[dead] = new_action_matrix[dead]


def finish_games(games, x_players, o_players):
  while True:
    moves = x_players.play(games.boards)
    games.update(moves, PLAYERS.X)
    moves = o_players.play(games.boards)
    games.update(moves, PLAYERS.O)
    if torch.all(games.game_over):
      break
  x_players.credits[games.winners == PLAYERS.X] += 1
  x_players.credits[games.losers == PLAYERS.X] -= 1
  o_players.credits[games.winners == PLAYERS.O] += 1
  o_players.credits[games.losers == PLAYERS.O] -= 1




credits = INIT_CREDS * torch.ones((BATCH_SIZE*2,), dtype=torch.int8)
action_matrix = torch.zeros((BATCH_SIZE*2, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)
players = Players(credits, action_matrix)
while True:
  t0 = time.time()
  games = Games()
  indices = torch.randperm(BATCH_SIZE*2)
  x_players = Players(players.credits[indices][:BATCH_SIZE], players.action_matrix[indices][:BATCH_SIZE])
  o_players = Players(players.credits[indices][BATCH_SIZE:], players.action_matrix[indices][BATCH_SIZE:])
  t1 = time.time()
  finish_games(games, x_players, o_players)
  t2 = time.time()
  players = Players(torch.cat([x_players.credits, o_players.credits]), torch.cat([x_players.action_matrix, o_players.action_matrix]))
  players.mate()
  t3 = time.time()
  print(t1-t0, t2-t1, t3-t2)
  print(games.total_moves, x_players.credits.float().mean(), o_players.credits.float().mean())
 