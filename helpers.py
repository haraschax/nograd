class PLAYERS:
  NONE = 0
  X = 1
  O = 2

BOARD_ROWS, BOARD_COLS = 6, 7
BOARD_SIZE = BOARD_ROWS * BOARD_COLS

def next_player(player):
  return PLAYERS.O if player == PLAYERS.X else PLAYERS.X