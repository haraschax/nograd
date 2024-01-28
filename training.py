from helpers import PLAYERS, next_player
from tqdm import tqdm
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


###### GET OPTIMAL MOVES ######

def is_winner(board, player):
  return (np.any(np.all(board == player, axis=1)) or \
     np.any(np.all(board == player, axis=0)) or \
     np.all(np.diag(board) == player) or \
     np.all(np.diag(np.fliplr(board)) == player))

def is_draw(board):
  return not np.any(board == 0) 

def generate_valid_boards(current_board, player, all_boards, board_hashes):
  board_hash = current_board.tobytes()
  if board_hash in board_hashes:
    return
  if is_winner(current_board, PLAYERS.X) or is_winner(current_board, PLAYERS.O) or is_draw(current_board):
    return

  board_hashes.add(board_hash)
  all_boards.append(current_board.copy())
  for i in range(3):
    for j in range(3):
      if current_board[i, j] == 0:
        current_board[i, j] = player
        generate_valid_boards(current_board, next_player(player), all_boards, board_hashes)
        current_board[i, j] = 0


def get_optimal_move(board, player):
  assert not is_winner(board, PLAYERS.X)
  assert not is_winner(board, PLAYERS.O)
  assert not is_draw(board)

  best_score = -np.inf
  best_move = None

  for i in range(3):
    for j in range(3):
      if board[i, j] == PLAYERS.NONE:
        board[i, j] = player
        if is_winner(board, player):
          score = 1
        elif is_draw(board):
          score = 0
        else:
          _, score = get_optimal_move(board, next_player(player))
          # Invert the score as it is from the opponent's perspective
          score = -score 
        if score > best_score:
          best_score = score
          best_move = (i, j)
        board[i, j] = PLAYERS.NONE
  return best_move, best_score


def generate_perfect_moves():
  all_boards = []
  board_hashes = set()
  initial_board = np.zeros((3, 3), dtype=int)
  generate_valid_boards(initial_board, PLAYERS.X, all_boards, board_hashes)
  board_move_pairs = []
  for board in tqdm(all_boards):
    player = PLAYERS.O if np.sum(board == PLAYERS.X) > np.sum(board == PLAYERS.O) else PLAYERS.X
    move, _ = get_optimal_move(board, player)
    board_move_pairs.append((board, move))
  return board_move_pairs


###### TRAINING STUFF ######

class TicTacToeDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        board, move = self.data[idx]
        board = torch.tensor(board).long()
        board_flat_onehot = torch.nn.functional.one_hot(board.flatten(), num_classes=3).float()
        
        board = torch.cat([board_flat_onehot.flatten(), 0.5*torch.ones((9,))])
        move = move[0] * 3 + move[1]  # Convert move to single integer
        return board, move


class TicTacToeNN(nn.Module):
    def __init__(self):
        super(TicTacToeNN, self).__init__()
        self.fc1 = nn.Linear(9*4, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 9, bias=False)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x



##### TRAIN PERFECT PLAYER #####

print('Generating perfect moves...')
dataset = TicTacToeDataset(generate_perfect_moves())
data_loader = DataLoader(dataset, batch_size=128, shuffle=True)
print('Starting training...')
model = TicTacToeNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

steps = 1000
pbar = tqdm(range(steps))
for step in pbar:
    total_loss = 0
    for board, optimal_move in data_loader:
        outputs = model(board)
        loss = criterion(outputs, optimal_move)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    pbar.set_description(f'Epoch {step+1}/{steps}, Loss: {total_loss/len(data_loader)}')
good_weights = []
for param in model.parameters():
    good_weights.append(param.data)
print('Saving weights...')
with open('perfect_player_weights.pkl', 'wb') as f:
  pickle.dump(good_weights, f)
