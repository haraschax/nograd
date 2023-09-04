
import random
import torch
import numpy as np
import torch.nn as nn

DNA_SIZE = 200
NOISE_SIZE = 100
BOARD_SIZE = 9
ACTION_SIZE = 6
INIT_CREDS = 5
WEIGHT_SIZE = 61906

dna_unroll_model = nn.Linear(DNA_SIZE, WEIGHT_SIZE)


class Agent():
  def __init__(self, dna):
    self.dna = dna
    self.hash = random.getrandbits(128)
    self.action_model = nn.Sequential(nn.Linear(BOARD_SIZE + NOISE_SIZE, 100),
                                      nn.ReLU(),
                                      nn.Linear(100, ACTION_SIZE))
    self.birth_model = nn.Sequential(nn.Linear(DNA_SIZE + NOISE_SIZE, 100),
                                     nn.ReLU(),
                                     nn.Linear(100, DNA_SIZE))
    self.credits = INIT_CREDS

    weights_idx = 0
    all_weights = dna_unroll_model(dna)
    for param in self.action_model.parameters():
      param.data = nn.parameter.Parameter(all_weights[weights_idx:weights_idx + param.flatten().shape[0]].reshape(param.shape))
      weights_idx += param.flatten().shape[0]
    for param in self.birth_model.parameters():
      param.data = nn.parameter.Parameter(all_weights[weights_idx:weights_idx + param.flatten().shape[0]].reshape(param.shape))
      weights_idx += param.flatten().shape[0]
    #print(f"Agent {self.hash} has {weights_idx} weights")


  def act(self, board):
    state = torch.from_numpy(board).float().flatten()
    state_with_noise = torch.cat([state, torch.randn(NOISE_SIZE)])

    action = self.action_model(state_with_noise)
    x = np.argmax(action.detach().numpy()[:3])
    y = np.argmax(action.detach().numpy()[3:])
    return x,y
  
  def give_birth(self,):
    dna_with_noise = torch.cat([self.dna, torch.randn(NOISE_SIZE)])

    new_dna = self.birth_model(dna_with_noise)
    self.credits -= INIT_CREDS
    return new_dna
  


dummy_agent = Agent(torch.randn(DNA_SIZE))
total_size = 0
for param in dummy_agent.action_model.parameters():
  total_size += param.flatten().shape[0]
for param in dummy_agent.birth_model.parameters():
  total_size += param.flatten().shape[0]
print(f"Total size of models: {total_size}")

dna_unroll_model = nn.Linear(DNA_SIZE, WEIGHT_SIZE)

