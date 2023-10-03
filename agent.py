
import random
import torch
import numpy as np
import torch.nn as nn


NOISE_SIZE = 10
STATE_SIZE = 50
BOARD_SIZE = 18
ACTION_SIZE = 9
INIT_CREDS = 5
EMBED_N = 30
ACTION_WEIGHTS_SIZE = (BOARD_SIZE + NOISE_SIZE) * EMBED_N + EMBED_N * EMBED_N + EMBED_N * (ACTION_SIZE + STATE_SIZE) + EMBED_N + EMBED_N + ACTION_SIZE + STATE_SIZE
BIRTH_WEIGHTS_SIZE = EMBED_N * EMBED_N + EMBED_N * EMBED_N + EMBED_N * ACTION_WEIGHTS_SIZE + EMBED_N + EMBED_N + ACTION_WEIGHTS_SIZE
DNA_SIZE = ACTION_WEIGHTS_SIZE + BIRTH_WEIGHTS_SIZE




class Agent():
  def __init__(self, dna=None):
    if dna is None:
      self.dna = torch.zeros(DNA_SIZE)
    else:
      self.dna = dna

    self.hash = random.getrandbits(128)
    self.action_model = nn.Sequential(nn.Linear(BOARD_SIZE + NOISE_SIZE, EMBED_N),
                                      nn.ReLU(),
                                      nn.Linear(EMBED_N, EMBED_N),
                                      nn.ReLU(),
                                      nn.Linear(EMBED_N, ACTION_SIZE + STATE_SIZE))
    ACTION_WEIGHTS_SIZE = (BOARD_SIZE + NOISE_SIZE) * EMBED_N + EMBED_N * EMBED_N + EMBED_N * (ACTION_SIZE + STATE_SIZE) + EMBED_N + EMBED_N + ACTION_SIZE + STATE_SIZE
    self.birth_model = nn.Sequential(nn.Linear(EMBED_N, EMBED_N),
                                      nn.ReLU(),
                                      nn.Linear(EMBED_N, EMBED_N),
                                      nn.ReLU(),
                                      nn.Linear(EMBED_N, ACTION_WEIGHTS_SIZE))

    self.credits = INIT_CREDS
    self.state = torch.zeros(STATE_SIZE)

    weights_idx = 0
    all_weights = self.dna
    weights_idx = 0
    for param in self.birth_model.parameters():
      param.data = nn.parameter.Parameter(all_weights[weights_idx:weights_idx + param.flatten().shape[0]].reshape(param.shape))
      weights_idx += param.flatten().shape[0]
    assert weights_idx == BIRTH_WEIGHTS_SIZE

    for param in self.action_model.parameters():
      param.data = nn.parameter.Parameter(all_weights[weights_idx:weights_idx + param.flatten().shape[0]].reshape(param.shape))
      weights_idx += param.flatten().shape[0]
    assert weights_idx == DNA_SIZE


  def act(self, board):
    board = torch.from_numpy(board).float().flatten()
    board_better = torch.cat([board == -1, board == 1]).float()
    state_with_noise = torch.cat([board_better, 0*torch.randn(NOISE_SIZE)])

    out = self.action_model(state_with_noise)
    action = out[:ACTION_SIZE]

    if np.isnan(action.detach().numpy()).any():
      return np.nan, np.nan
    else:
      x = np.argmax(action.detach().numpy()) // 3
      y = np.argmax(action.detach().numpy()) % 3
      return x,y
  
  def give_birth(self,):
    noise = torch.randn(EMBED_N)
    action_weights_change = self.birth_model(noise)
    birth_weights, action_weights = self.dna[:BIRTH_WEIGHTS_SIZE], self.dna[BIRTH_WEIGHTS_SIZE:]

    new_dna = torch.cat([birth_weights + torch.randn(birth_weights.shape), action_weights_change])

    self.credits -= INIT_CREDS
    return new_dna