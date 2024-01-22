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
  generate_perfect_moves()

# Example usage:
#initial_board = np.zeros((3, 3), dtype=int)
#print(get_optimal_move(initial_board, -1))  # Start with X (-1)