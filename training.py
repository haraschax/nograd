 
import torch
import torch.nn as nn
import torch.optim as optim

# Assume `data` is a list of (board, optimal_move) pairs.
# Assume `board` is a flattened 1x9 tensor and `optimal_move` is an integer [0-8].


import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TicTacToeDataset(Dataset):
    def __init__(self, data):
        # data: a list of (board, move) pairs
        # board: a 3x3 numpy array
        # move: a (row, column) tuple
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        board, move = self.data[idx]
        board = board.flatten()
        boards_onehot = np.zeros((9, 3))
        boards_onehot[:,0] = (board == 0)
        boards_onehot[:,1] = (board == -1)
        boards_onehot[:,2] = (board == 1)

        board = boards_onehot.flatten().astype(np.float32)
        
        board = np.concatenate([board, torch.randn((9,))])
        board = torch.tensor(board, dtype=torch.float32)  # Convert board to tensor and flatten
        move = move[0] * 3 + move[1]  # Convert move to single integer
        return board, move

# Example usage:

# Assume `data` is your list of (board, move) pairs.
# Example data for illustration:
# data = [(np.array([[-1, 0, 1], [1, -1, 0], [0, 1, -1]]), (1, 1)), ...]
# where -1 is 'X', 1 is 'O', and 0 is an empty cell.

# Create dataset and dataloader
import pickle
dataset = TicTacToeDataset(pickle.load(open('move_pairs.pkl', 'rb')))
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)



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

# Instantiate the model, loss function and optimizer
model = TicTacToeNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 1000
for epoch in range(epochs):
    total_loss = 0
    
    for board, optimal_move in data_loader:
        board, optimal_move = torch.tensor(board).float(), torch.tensor(optimal_move).long()
        
        # Forward pass
        outputs = model(board)
        loss = criterion(outputs, optimal_move)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data_loader)}')
good_weights = []
for param in model.parameters():
    good_weights.append(param.data)
    print(good_weights[-1].shape)
with open('good_weights.pkl', 'wb') as f:
  pickle.dump(good_weights, f)
