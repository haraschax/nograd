from batch_arena import Games, PLAYERS, EMBED_N, BOARD_SIZE, DEVICE, Players, INIT_CREDS, BOARD_COLS, BOARD_ROWS

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import random
import time
import torch
import pickle

# Initialize Pygame
pygame.init()

SQUARE_SIZE = 100
WIDTH, HEIGHT = BOARD_ROWS * SQUARE_SIZE, BOARD_COLS * SQUARE_SIZE
LINE_WIDTH = 10
MARK_SIZE = 70
CROSS_SIZE = 30
CIRCLE_RADIUS = MARK_SIZE // 2
CIRCLE_WIDTH = 10
CROSS_WIDTH = 25
BATCH_SIZE = 1

BG_COLOR = (28, 170, 156)
LINE_COLOR = (23, 145, 135)
CIRCLE_COLOR = (239, 231, 200)
CROSS_COLOR = (66, 66, 66)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Tic Tac Toe')
screen.fill(BG_COLOR)

def draw_lines():
  for i in range(1, BOARD_ROWS):
    pygame.draw.line(screen, LINE_COLOR, (0, i * SQUARE_SIZE), (WIDTH, i * SQUARE_SIZE), LINE_WIDTH)
  for i in range(1, BOARD_COLS):
    pygame.draw.line(screen, LINE_COLOR, (i * SQUARE_SIZE, 0), (i * SQUARE_SIZE, HEIGHT), LINE_WIDTH)

def draw_figures(board):
  for row in range(BOARD_ROWS):
    for col in range(BOARD_COLS):
      center_x, center_y = int(col * SQUARE_SIZE + SQUARE_SIZE/2), int(row * SQUARE_SIZE + SQUARE_SIZE/2)
      if board[row][col] == PLAYERS.O:
        pygame.draw.circle(screen, CIRCLE_COLOR, (center_x, center_y), CIRCLE_RADIUS, CIRCLE_WIDTH)
      if board[row][col] == PLAYERS.X:
        pygame.draw.line(screen, CROSS_COLOR, (center_x - CROSS_SIZE, center_y - CROSS_SIZE),
                                              (center_x + CROSS_SIZE, center_y + CROSS_SIZE), CROSS_WIDTH)  
        pygame.draw.line(screen, CROSS_COLOR, (center_x - CROSS_SIZE, center_y + CROSS_SIZE),
                                              (center_x + CROSS_SIZE, center_y - CROSS_SIZE), CROSS_WIDTH)



if __name__ == '__main__':
  draw_lines()
  games = Games(bs=1)

  BATCH_SIZE = 1
  good_weights = pickle.load(open('perfect_player_weights.pkl', 'rb'))
  golden_params = {}
  golden_params['input'] = torch.cat([good_weights[0].T[None,:,:].to(device=DEVICE)  for _ in range(BATCH_SIZE)], dim=0)
  golden_params['bias'] = torch.cat([good_weights[1][None,:].to(device=DEVICE) for _ in range(BATCH_SIZE)], dim=0)
  golden_params['output'] = torch.cat([good_weights[2].T[None,:,:].to(device=DEVICE) for _ in range(BATCH_SIZE)], dim=0)
  players = Players(golden_params)

  while not games.game_over[0]:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      if event.type == pygame.MOUSEBUTTONDOWN:
        mouseX = event.pos[0]
        mouseY = event.pos[1]

        clicked_row = int(mouseY // SQUARE_SIZE)
        clicked_col = int(mouseX // SQUARE_SIZE)
        move = torch.zeros((3,3), device=DEVICE)
        
        move[clicked_row, clicked_col] = 1
        games.update(move.reshape((1,BOARD_SIZE)), PLAYERS.X)
        if not games.game_over[0]:
          move = players.play(games.boards, test=True)
          games.update(move.reshape((1,BOARD_SIZE)), PLAYERS.O)

    draw_figures(games.boards.cpu().reshape((BOARD_ROWS,BOARD_COLS)).numpy())
    pygame.display.update()
    time.sleep(0.05)
  if games.winners[0] == PLAYERS.X:
    print("X wins")
  elif games.winners[0] == PLAYERS.O:
    print("O wins")
  else:
    print("Nobody wins")
  time.sleep(1)
  pygame.quit()
