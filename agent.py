
import random
import torch
torch.set_grad_enabled(False)
import numpy as np
import torch.nn as nn


NOISE_SIZE = 10
STATE_SIZE = 1
BOARD_SIZE = 18
ACTION_SIZE = 9
INIT_CREDS = 5
EMBED_N = 128
META_SIZE = 2
ACTION_WEIGHTS_SIZE = (BOARD_SIZE + NOISE_SIZE) * EMBED_N + EMBED_N * (ACTION_SIZE + STATE_SIZE) + EMBED_N + ACTION_SIZE + STATE_SIZE
BIRTH_WEIGHTS_SIZE = NOISE_SIZE * EMBED_N + EMBED_N + EMBED_N * META_SIZE + META_SIZE
DNA_SIZE = ACTION_WEIGHTS_SIZE + BIRTH_WEIGHTS_SIZE




class Agent():
  def __init__(self, dna=None):
    if dna is None:
      self.dna = torch.randn(DNA_SIZE)
      import pickle
      good_weights = pickle.load(open('good_weights.pkl', 'rb'))
      assert len(good_weights) == ACTION_WEIGHTS_SIZE
      #self.dna[BIRTH_WEIGHTS_SIZE:] = good_weights + torch.randn(ACTION_WEIGHTS_SIZE)*0.1
    else:
      self.dna = dna

    self.hash = random.getrandbits(128)
    self.action_model = nn.Sequential(nn.Linear(BOARD_SIZE + NOISE_SIZE, EMBED_N),
                                      nn.ReLU(),
                                      nn.Linear(EMBED_N, ACTION_SIZE + STATE_SIZE)).to(device='cuda')
    self.birth_model = nn.Sequential(nn.Linear(NOISE_SIZE, EMBED_N),
                                      nn.ReLU(),
                                      nn.Linear(EMBED_N, META_SIZE)).to(device='cuda')

    self.credits = INIT_CREDS
    self.state = torch.zeros(STATE_SIZE)
    self.births = 0

    weights_idx = 0
    all_weights = self.dna
    weights_idx = 0
    self.mutation_rate = 0 
    self.mutation_size = 0

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
    state_with_noise = torch.cat([board_better, torch.randn(NOISE_SIZE)])

    out = self.action_model(state_with_noise)
    action = out[:ACTION_SIZE]
    if np.isnan(action.detach().numpy()).any():
      return np.nan, np.nan
    else:
      x = np.argmax(action.detach().numpy()) // 3
      y = np.argmax(action.detach().numpy()) % 3
      if torch.rand(1) < 0.0:
        x = np.random.randint(3)
        y = np.random.randint(3)
      return x,y
  
  def give_birth(self,):
    noise = torch.randn(NOISE_SIZE)
    mutate_meta = self.birth_model(0*noise)
    #birth_weights, action_weights = self.dna[:BIRTH_WEIGHTS_SIZE], self.dna[BIRTH_WEIGHTS_SIZE:] + action_weights_change

    #new_dna = torch.cat([birth_weights *10 ** (0.1 * torch.randn(birth_weights.shape)), action_weights])
    #mutation_rate = 10**self.dna[-1]
    #print(mutation_rate)
    self.mutation_rate = 1e-3 #1.2**mutate_meta[0]
    self.mutation_size = 1 #1.2**mutate_meta[1]
    mutation = self.mutation_size * (torch.rand(self.dna.shape) < self.mutation_rate)
    new_dna = self.dna + mutation * torch.randn(self.dna.shape)
    self.births += 1

    self.credits -= INIT_CREDS
    return new_dna
