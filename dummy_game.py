import numpy as np

class DummyGame():
  def __init__(self):
    self.winner = 0
    self.last_player = 0
    self.total_moves = 0
    self.board = np.array([[0,0,0],[0,0,0],[0,0,0]])

  
  def move(self, x, y, player):
    #print(f"Player {player} moves to ({x}, {y}), self.last_player = {self.last_player}")
    assert player == 1 or player == -1
    assert self.last_player != player
    self.last_player = player
    if x < 1 or y < 1:
      self.winner = -player
    else:
      self.total_moves += 1

  @property
  def isover(self):
    return self.winner != 0 or self.total_moves >= 20
  