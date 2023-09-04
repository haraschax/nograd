import numpy as np

class TicTacToe():
  def __init__(self):
    self.board = np.array([[0,0,0],[0,0,0],[0,0,0]])
    self.winner = 0
    self.last_player = 0
    self.credits = 5

  def check_winner(self):
    for i in range(3):
      if self.board[i][0] == self.board[i][1] == self.board[i][2] != 0:
        self.winner = self.board[i][0]
      if self.board[0][i] == self.board[1][i] == self.board[2][i] != 0:
        self.winner = self.board[0][i]
    if self.board[0][0] == self.board[1][1] == self.board[2][2] != 0:
      self.winner = self.board[0][0]
  
  def move(self, x, y, player):
    #print(f"Player {player} moves to ({x}, {y}), self.last_player = {self.last_player}")
    assert player == 1 or player == -1
    assert self.last_player != player
    self.last_player = player
    if self.board[x][y] == 0:
      self.board[x][y] = player
      self.check_winner()
    else:
      self.winner = -player

  def board_full(self):
    for i in range(3):
      for j in range(3):
        if self.board[i][j] == 0:
          return False
    return True
  
  def __str__(self) -> str:
    line1 = f"{self.board[0][0]}|{self.board[0][1]}|{self.board[0][2]}"
    line2 = f"{self.board[1][0]}|{self.board[1][1]}|{self.board[1][2]}"
    line3 = f"{self.board[2][0]}|{self.board[2][1]}|{self.board[2][2]}"
    return f"{line1}\n{line2}\n{line3}"

  @property
  def isover(self):
    return self.winner != 0 or self.board_full()
  
  @property
  def total_moves(self):
    return np.sum(np.abs(self.board))