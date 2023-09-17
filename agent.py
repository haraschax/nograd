
import random
import torch
import numpy as np
import torch.nn as nn

DNA_SIZE = 50
NOISE_SIZE = 50
STATE_SIZE = 50
BOARD_SIZE = 18
ACTION_SIZE = 9
INIT_CREDS = 5
WEIGHT_SIZE = 10000
EMBED_N = 20

dna_unroll_model = nn.Sequential(nn.Linear(DNA_SIZE, WEIGHT_SIZE),
                                 nn.LayerNorm(WEIGHT_SIZE))
dna_unroll_model = nn.Linear(DNA_SIZE, WEIGHT_SIZE)
noise = torch.randn(DNA_SIZE)

class Agent():
  def __init__(self, dna):
    self.dna = torch.tensor(dna)
    self.hash = random.getrandbits(128)
    self.action_model = nn.Sequential(nn.Linear(BOARD_SIZE + NOISE_SIZE + STATE_SIZE, EMBED_N),
                                      nn.ReLU(),
                                      nn.Linear(EMBED_N, EMBED_N),
                                      nn.ReLU(),
                                      nn.Linear(EMBED_N, ACTION_SIZE + STATE_SIZE))
    
    self.birth_model = nn.Sequential(nn.Linear(DNA_SIZE + NOISE_SIZE + STATE_SIZE, EMBED_N),
                                      nn.ReLU(),
                                      nn.Linear(EMBED_N, DNA_SIZE))

    self.credits = INIT_CREDS
    self.state = torch.zeros(STATE_SIZE)

    weights_idx = 0
    #assert not (np.isnan(dna).any())

    all_weights = dna_unroll_model(torch.tensor(dna))
    #assert not (np.isnan(all_weights.detach().numpy()).any())
    #all_weights = torch.outer(self.dna, self.dna).flatten()
    #all_weights = self.dna
    #assert len(all_weights) >= WEIGHT_SIZE
    for param in self.action_model.parameters():
      param.data = nn.parameter.Parameter(all_weights[weights_idx:weights_idx + param.flatten().shape[0]].reshape(param.shape))
      weights_idx += param.flatten().shape[0]
    for param in self.birth_model.parameters():
      param.data = nn.parameter.Parameter(all_weights[weights_idx:weights_idx + param.flatten().shape[0]].reshape(param.shape))
      weights_idx += param.flatten().shape[0] 
    #print(f"Agent {self.hash} has {weights_idx} weights")


  def act(self, board):
    board = torch.from_numpy(board).float().flatten()
    board_better = torch.cat([board == -1, board == 1])
    state_with_noise = torch.cat([board_better, torch.randn(NOISE_SIZE), self.state])

    out = self.action_model(state_with_noise)
    action = out[:ACTION_SIZE]
    #assert not ((np.isnan(self.state.detach().numpy()).any()))
    self.state = out[ACTION_SIZE:]

    if np.isnan(action.detach().numpy()).any():
      return np.nan, np.nan
    else:
      x = np.argmax(action.detach().numpy()) // 3
      y = np.argmax(action.detach().numpy()) % 3
      return x,y
  
  def give_birth(self,):
    noise = torch.randn(NOISE_SIZE)
    dna_with_noise = torch.cat([self.dna, noise, self.state])
    #assert not ((np.isnan(self.state.detach().numpy()).any()))
    #assert not ((np.isnan(dna_with_noise.detach().numpy()).any()))

    #print(np.linalg.norm(self.dna), np.linalg.norm(self.birth_model(dna_with_noise).detach().numpy()))
    #print(False)
    new_dna = self.dna + noise
    #print(self.dna, new_dna)
    #assert False
    #new_dna = self.dna + noise
    #new_dna = self.dna + noise

    self.credits -= INIT_CREDS
    return new_dna.detach().numpy()
  


dummy_agent = Agent(torch.randn(DNA_SIZE))
total_size = 0
for param in dummy_agent.action_model.parameters():
  total_size += param.flatten().shape[0]
for param in dummy_agent.birth_model.parameters():
  total_size += param.flatten().shape[0]
print(f"Total size of models: {total_size}")

dna_unroll_model = nn.Linear(DNA_SIZE, WEIGHT_SIZE)

