import numpy as np

class TicTacToe():
  def __init__(self):
    self.board = np.array([[0,0,0],[0,0,0],[0,0,0]])
    self.winner = 0
    self.last_player = 0

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
    if np.isfinite(x) and np.isfinite(y) and self.board[x][y] == 0:
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
  












import numpy as np

def is_winner(board, player):
    if np.any(np.all(board == player, axis=1)) or \
       np.any(np.all(board == player, axis=0)) or \
       np.all(np.diag(board) == player) or \
       np.all(np.diag(np.fliplr(board)) == player):
        return True
    return False

def is_draw(board):
    return not np.any(board == 0) 

def generate_valid_boards(current_board, player, all_boards, seen_boards):
    board_hash = current_board.tobytes()
    if board_hash in seen_boards:
        return
    if is_winner(current_board, -1) or is_winner(current_board, 1) or is_draw(current_board):
        return
    
    seen_boards.add(board_hash)
    all_boards.append(current_board.copy())
    for i in range(3):
        for j in range(3):
            if current_board[i, j] == 0:
                current_board[i, j] = player
                generate_valid_boards(current_board, -player, all_boards, seen_boards)
                current_board[i, j] = 0  # Backtrack

# Example usage:
all_boards = []
seen_boards = set()
initial_board = np.zeros((3, 3), dtype=int)
generate_valid_boards(initial_board, -1, all_boards, seen_boards)  # Start with X (-1)


def get_optimal_move(board, player):
    if is_winner(board, player):
        return None, 1  # Already won, no move necessary
    if is_draw(board):
        return None, 0  # Draw, no move necessary
    
    best_score = -float('inf')
    best_move = None

    for i in range(3):
        for j in range(3):
            if board[i, j] == 0:  # If the spot is empty
                board[i, j] = player  # Make the move
                
                if is_winner(board, player):  # If the move leads to a win
                    score = 1
                elif is_draw(board):  # If the move leads to a draw
                    score = 0
                else:  # Otherwise, the opponent makes their optimal move
                    _, score = get_optimal_move(board, -player)
                    score = -score  # Invert the score as it is from the opponent's perspective
                
                if score > best_score:  # If the move is better than current best
                    best_score = score
                    best_move = (i, j)
                
                board[i, j] = 0  # Undo the move
    
    return best_move, best_score

import pickle

def generate_perfect_moves():
  from tqdm import tqdm
  board_move_pairs = []
  for board in tqdm(all_boards):
      if np.sum(board) == 0:
         player = -1
      else:
          player = 1
      move, score = get_optimal_move(board, player)
      board_move_pairs.append((board, move))
  with open('move_pairs.pkl', 'wb') as f:
    pickle.dump(board_move_pairs, f)

if __name__ == "__main__":
  board = np.zeros((3, 3), dtype=int)
  player = -1
  board_move_pairs = pickle.load(open('move_pairs.pkl', 'rb'))
  while not is_winner(board, -1) and not is_winner(board, 1) and not is_draw(board):
    for b, m in board_move_pairs:
      if np.all(b == board):
        board[m[0], m[1]] = player
        player = -player
        break
    print(board)
    location = input("Please input location of your move: \n").split(',')
    board[int(location[0]), int(location[1])] = player
    player = -player


# Example usage:
#initial_board = np.zeros((3, 3), dtype=int)
#print(get_optimal_move(initial_board, -1))  # Start with X (-1)