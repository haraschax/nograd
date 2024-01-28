
import torch
import time
from helpers import PLAYERS, next_player, BOARD_SIZE
import torch.nn.functional as F

MAX_MOVES = 10
BATCH_SIZE = 10000
INIT_CREDS = 2
EMBED_N = 128
NOISE_SIZE = 9
import pickle

DEVICE = 'cuda'

class PLAYERS:
  NONE = 0
  X = 1
  O = 2


class Games():
  def __init__(self, bs=BATCH_SIZE):
    self.bs = bs
    self.boards = torch.zeros((self.bs, BOARD_SIZE), dtype=torch.int8, device=DEVICE)
    self.winners = torch.zeros((self.bs,), dtype=torch.int8, device=DEVICE)
    self.update_game_over()


  def update(self, moves, player):
    assert len(moves) == self.bs
    assert len(moves) == self.boards.shape[0]

    move_idxs = torch.argmax(moves, dim=1, keepdim=True)
    
    illegal_movers = self.boards.gather(1, move_idxs) != PLAYERS.NONE
    self.winners[(illegal_movers[:,0]) & (self.game_over == 0)] = PLAYERS.O if player == PLAYERS.X else PLAYERS.X

    move_scattered = torch.zeros_like(self.boards.to(dtype=torch.bool))
    move_scattered.scatter_(1, move_idxs, 1)

    self.boards = self.boards + (self.game_over == 0)[:,None] * move_scattered * player
    self.check_winners()
    self.update_game_over()

  def check_winners(self):
    boards = self.boards.reshape((-1, 3, 3))
    M, rows, cols = boards.shape
    assert rows == 3 and cols == 3, "Each board must be a 3x3 grid."
    winners = torch.zeros(M, dtype=torch.int8, device=DEVICE)
    for player in [PLAYERS.X, PLAYERS.O]:
      rows_winner = torch.any(torch.all(boards == player, dim=1), dim=1)
      cols_winner = torch.any(torch.all(boards == player, dim=2), dim=1)
      winners[rows_winner | cols_winner] = player

      diag1 = boards[:, torch.arange(3), torch.arange(3)]
      diag2 = boards[:, torch.arange(3), torch.arange(2, -1, -1)]
      diag1_winner = torch.all(diag1 == player, dim=1)
      diag2_winner = torch.all(diag2 == player, dim=1)
      winners[diag1_winner | diag2_winner] = player
    
    self.winners[self.winners == PLAYERS.NONE] = winners[self.winners == PLAYERS.NONE]

  @property
  def losers(self):
    losers = torch.zeros_like(self.winners)
    losers[self.winners == PLAYERS.X] = PLAYERS.O
    losers[self.winners == PLAYERS.O] = PLAYERS.X
    return losers
  
  @property
  def total_moves(self):
    return (self.boards != PLAYERS.NONE).sum(dim=1).float().mean()

  def update_game_over(self):
    self.game_over = (self.winners != PLAYERS.NONE) | ((self.boards != PLAYERS.NONE).sum(dim=1) == BOARD_SIZE)


class Players():
  def __init__(self, params, credits=None):
    self.params = params
    self.credits = credits

  def play(self, boards, test=False):
    boards_onehot = torch.zeros((boards.shape[0], BOARD_SIZE, 3), dtype=torch.float32, device=DEVICE)
    boards_onehot[:,:,0] = boards == PLAYERS.NONE
    boards_onehot[:,:,1] = boards == PLAYERS.X
    boards_onehot[:,:,2] = boards == PLAYERS.O
    if test:
      noise = 0.5*torch.ones_like(boards.float())
    else:
      noise = torch.randn_like(boards.float())
    inputs = torch.cat([boards_onehot.reshape((-1, BOARD_SIZE*3)), noise], dim=1)
    embed = torch.einsum('bji, bj->bi', self.params['input'], inputs)
    embed = embed + self.params['bias']
    embed = torch.relu(embed)
    moves = torch.einsum('bji, bj->bi', self.params['output'], embed)
    moves = torch.softmax(moves, dim=1)
    return moves
  
  def mate(self, mutation_rate=1e-4):
    assert self.credits is not None, "Credits must be set before mating."
    mutation_rate = torch.exp(self.params['mutuation'].sum(dim=1))
    dead = (self.credits == 0).nonzero(as_tuple=True)[0]
    can_mate = (self.credits > INIT_CREDS).nonzero(as_tuple=True)[0]
    assert len(can_mate) >= len(dead)
    self.credits[dead] = INIT_CREDS
    self.credits[can_mate[:len(dead)]] -= INIT_CREDS
    for key in self.params:
      if key in ['block']:
        mutation_rate_full = mutation_rate.reshape((-1, 1, 1,1)).repeat((1, self.params[key].shape[1], self.params[key].shape[2], self.params[key].shape[3]))
 
      elif key in ['input', 'output', 'block_bias']:
        mutation_rate_full = mutation_rate.reshape((-1, 1, 1)).repeat((1, self.params[key].shape[1], self.params[key].shape[2]))
      else:
        mutation_rate_full = mutation_rate.reshape((-1, 1)).repeat((1, self.params[key].shape[1]))
      param = torch.clone(self.params[key])
      mutation = (torch.rand_like(param) < mutation_rate_full).float()
      new_param = (1 - mutation) * param + mutation * (torch.randn_like(param))
      self.params[key][dead] = new_param[can_mate[:len(dead)]]

  def avg_log_mutuation(self):
    return self.params['mutuation'].sum(dim=1).float().mean().item()

def finish_games(games, x_players, o_players, test=False):
  while True:
    moves = x_players.play(games.boards, test=test)
    games.update(moves, PLAYERS.X)
    if torch.all(games.game_over):
      break
    moves = o_players.play(games.boards, test=test)
    games.update(moves, PLAYERS.O)
    if torch.all(games.game_over):
      break
  if not test:
    x_players.credits[games.winners == PLAYERS.X] += 1
    x_players.credits[games.losers == PLAYERS.X] -= 1
    o_players.credits[games.winners == PLAYERS.O] += 1
    o_players.credits[games.losers == PLAYERS.O] -= 1


def splice_params(params, indices):
  new_params = {}
  for key in params:
    new_params[key] = params[key][indices]
  return new_params

def concat_params(params1, params2):
  new_params = {}
  for key in params1:
    new_params[key] = torch.cat([params1[key], params2[key]])
  return new_params



from tensorboardX import SummaryWriter

def train_run(name='', mutation_rate=1e-4):
  writer = SummaryWriter(f'runs/{name}')

  credits = INIT_CREDS * torch.ones((BATCH_SIZE*2,), dtype=torch.int8, device=DEVICE)

  params = {'input': torch.randn((BATCH_SIZE*2, BOARD_SIZE*4, EMBED_N), dtype=torch.float32, device=DEVICE),
            'bias': torch.randn((BATCH_SIZE*2, EMBED_N), dtype=torch.float32, device=DEVICE),
            'output': torch.randn((BATCH_SIZE*2, EMBED_N, BOARD_SIZE), dtype=torch.float32, device=DEVICE),
            'mutuation': torch.randn((BATCH_SIZE*2, 50), dtype=torch.float32, device=DEVICE)}
  
  import pickle
  good_weights = pickle.load(open('good_weights.pkl', 'rb'))
  golden_params = {}
  golden_params['input'] = torch.cat([good_weights[0].T[None,:,:].to(device=DEVICE)  for _ in range(BATCH_SIZE*2)], dim=0)
  golden_params['bias'] = torch.cat([good_weights[1][None,:].to(device=DEVICE) for _ in range(BATCH_SIZE*2)], dim=0)
  golden_params['output'] = torch.cat([good_weights[2].T[None,:,:].to(device=DEVICE) for _ in range(BATCH_SIZE*2)], dim=0)

  golden_players = Players(golden_params)


  players = Players(params, credits)
  t_start = time.time()
  for step in range(1000000):
    t0 = time.time()
    games = Games()
    indices = torch.randperm(BATCH_SIZE*2)
    x_players = Players(splice_params(players.params, indices[:BATCH_SIZE]), players.credits[indices][:BATCH_SIZE])
    o_players = Players(splice_params(players.params, indices[BATCH_SIZE:]), players.credits[indices][BATCH_SIZE:])
    t1 = time.time()
    finish_games(games, x_players, o_players)
    t2 = time.time()
    players = Players(concat_params(x_players.params, o_players.params), torch.cat([x_players.credits, o_players.credits]))
    players.mate(mutation_rate=mutation_rate)
    t3 = time.time()
    if step % 100 == 0:
      print(f'{step} games took {t3-t_start:.2f} seconds')
      print(t1-t0, t2-t1, t3-t2)
      print(f'Average total moves: {games.total_moves:.2f}, avg credits of X: {x_players.credits.float().mean():.2f}, avg credits of O: {o_players.credits.float().mean():.2f}')
      print(f'Average log mutuation: {players.avg_log_mutuation():.2e}')
      writer.add_scalar('total_moves', games.total_moves, step)
      writer.add_scalar('avg_log_mutuation', players.avg_log_mutuation(), step)
    if step % 1000 == 0:
      pickle.dump(players.params, open('player_params.pkl', 'wb'))
      games = Games(bs=BATCH_SIZE*2)
      finish_games(games, golden_players, players, test=True) 
      
      print(f'Vs perfect player avg total moves: {games.total_moves:.2f}, X win rate: {(games.winners == PLAYERS.X).float().mean():.2f}, O win rate: {(games.winners == PLAYERS.O).float().mean():.2f}')
      writer.add_scalar('perfect_total_moves', games.total_moves, step)
      writer.add_scalar('perfect_loss_rate',(games.winners == PLAYERS.X).float().mean(), step)
      writer.add_scalar('perfect_draw_rate',(games.winners == PLAYERS.NONE).float().mean(), step)

  writer.close()
  
if __name__ == '__main__':
  for i in range(5,7):
    mutation_rate = 10**(-i)
    name = f'run2_{i}'
    train_run(name=name, mutation_rate=mutation_rate)