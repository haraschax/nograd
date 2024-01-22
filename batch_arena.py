


import torch
import time

MAX_MOVES = 10
BATCH_SIZE = 20000
BOARD_ROWS, BOARD_COLS = 3, 3
BOARD_SIZE = BOARD_ROWS * BOARD_COLS
INIT_CREDS = 5
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
    #move_idxs = torch.multinomial(moves, num_samples=1)
    move_idxs = torch.argmax(moves, dim=1, keepdim=True)
  
    illegal_movers = self.boards.gather(1, move_idxs) != PLAYERS.NONE
    self.winners[illegal_movers[:,0]] = PLAYERS.O if player == PLAYERS.X else PLAYERS.X

    move_scattered = torch.zeros_like(self.boards.to(dtype=torch.bool))
    move_scattered.scatter_(1, move_idxs, 1)

    self.boards = self.boards + move_scattered * player
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
  def __init__(self, credits, params):
    self.credits = credits
    self.params = params

  def play(self, boards, test=False):
    boards_onehot = torch.zeros((boards.shape[0], BOARD_SIZE, 3), dtype=torch.float32, device=DEVICE)
    boards_onehot[:,:,0] = boards == PLAYERS.NONE
    boards_onehot[:,:,1] = boards == PLAYERS.X
    boards_onehot[:,:,2] = boards == PLAYERS.O
    inputs = torch.cat([boards_onehot.reshape((-1, BOARD_SIZE*3)), torch.rand_like(boards.float())], dim=1)
    embed = torch.einsum('bji, bj->bi', self.params['input'], inputs)
    embed = embed + self.params['bias']
    embed = torch.relu(embed)
    moves = torch.einsum('bji, bj->bi', self.params['output'], embed)
    moves = torch.softmax(moves, dim=1)
    if not test:
      random_moves = torch.rand(moves.shape[:1], device=DEVICE) < 1e-1
      moves = (1 - random_moves.float()[:,None]) * moves + random_moves[:,None] * torch.rand_like(moves)
    return moves
  
  def mate(self):
    mutation_rate = 1e-4
    dead = (self.credits == 0).nonzero(as_tuple=True)[0]
    can_mate = (self.credits > INIT_CREDS).nonzero(as_tuple=True)[0]
    assert len(can_mate) >= len(dead)
    self.credits[dead] = INIT_CREDS
    self.credits[can_mate[:len(dead)]] -= INIT_CREDS
    for key in self.params:
      param = torch.clone(self.params[key])
      mutation = (torch.rand_like(param) < mutation_rate).float()
      new_param = (1 - mutation) * param + mutation * (torch.randn_like(param))
      self.params[key][dead] = new_param[can_mate[:len(dead)]]


def finish_games(games, x_players, o_players, test=False):
  while True:
    moves = x_players.play(games.boards)
    games.update(moves, PLAYERS.X, test=test)
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

def __main__():
  writer = SummaryWriter()

  credits = INIT_CREDS * torch.ones((BATCH_SIZE*2,), dtype=torch.int8, device=DEVICE)

  params = {'input': torch.randn((BATCH_SIZE*2, BOARD_SIZE*4, EMBED_N), dtype=torch.float32, device=DEVICE),
            'bias': torch.randn((BATCH_SIZE*2, EMBED_N), dtype=torch.float32, device=DEVICE),
            'output': torch.randn((BATCH_SIZE*2, EMBED_N, BOARD_SIZE), dtype=torch.float32, device=DEVICE)}
  
  import pickle
  
  golden_params = {'input': torch.randn((BATCH_SIZE*2, BOARD_SIZE*4, EMBED_N), dtype=torch.float32, device=DEVICE),
                   'bias': torch.randn((BATCH_SIZE*2, EMBED_N), dtype=torch.float32, device=DEVICE),
                   'output': torch.randn((BATCH_SIZE*2, EMBED_N, BOARD_SIZE), dtype=torch.float32, device=DEVICE)}
  good_weights = pickle.load(open('good_weights.pkl', 'rb'))
  golden_params['input'][:] = good_weights[0].T
  golden_params['bias'][:] = good_weights[1]
  golden_params['output'][:] = good_weights[2].T

  golden_players = Players(torch.zeros((BATCH_SIZE*2,), dtype=torch.int8, device=DEVICE), golden_params)


  players = Players(credits, params)
  cnt = 0
  t_start = time.time()
  while True:
    t0 = time.time()
    games = Games()
    indices = torch.randperm(BATCH_SIZE*2)
    x_players = Players(players.credits[indices][:BATCH_SIZE], splice_params(players.params, indices[:BATCH_SIZE]))
    o_players = Players(players.credits[indices][BATCH_SIZE:], splice_params(players.params, indices[BATCH_SIZE:]))
    t1 = time.time()
    finish_games(games, x_players, o_players)
    t2 = time.time()
    players = Players(torch.cat([x_players.credits, o_players.credits]), concat_params(x_players.params, o_players.params))
    players.mate()
    t3 = time.time()
    #assert False
    if cnt % 100 == 0:
      #print(t1-t0, t2-t1, t3-t2)
      print(f'Average total moves: {games.total_moves:.2f}, avg credits of X: {x_players.credits.float().mean():.2f}, avg credits of O: {o_players.credits.float().mean():.2f}')
      #print(games.total_moves, x_players.credits.float().mean(), o_players.credits.float().mean())
    cnt += 1
    if cnt % 1000 == 0:
      pickle.dump(players.params, open('player_params.pkl', 'wb'))
      games = Games(bs=BATCH_SIZE*2)
      finish_games(games, golden_players, players, test=True)
      print(f'Vs perfect player avg total moves: {games.total_moves:.2f}, X win rate: {(games.winners == PLAYERS.X).float().mean():.2f}')
      
    #if games.total_moves > 7.5:
    #  print(f' {cnt} games took {t3-t_start:.2f} seconds to achieve 7.5')
    #  break
  
if __name__ == '__main__':
  __main__()
