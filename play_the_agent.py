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

# Constants
SQUARE_SIZE = 100
WIDTH, HEIGHT = BOARD_ROWS * SQUARE_SIZE, BOARD_COLS * SQUARE_SIZE
LINE_WIDTH = 10
MARK_SIZE = 70
CIRCLE_RADIUS = MARK_SIZE // 2
CIRCLE_WIDTH = 10
CROSS_WIDTH = 25
BATCH_SIZE = 1

# Colors
BG_COLOR = (28, 170, 156)
LINE_COLOR = (23, 145, 135)
CIRCLE_COLOR = (239, 231, 200)
CROSS_COLOR = (66, 66, 66)

# Screen Setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Tic Tac Toe')
screen.fill(BG_COLOR)

# Functions
def draw_lines():
  for i in range(1, BOARD_ROWS):
    pygame.draw.line(screen, LINE_COLOR, (0, i * SQUARE_SIZE), (WIDTH, i * SQUARE_SIZE), LINE_WIDTH)
  for i in range(1, BOARD_COLS):
    pygame.draw.line(screen, LINE_COLOR, (i * SQUARE_SIZE, 0), (i * SQUARE_SIZE, HEIGHT), LINE_WIDTH)

def draw_figures(board):
  for row in range(BOARD_ROWS):
    for col in range(BOARD_COLS):
      center_x, center_y = int(col * 100 + 100/2), int(row * 100 + 100/2)
      if board[row][col] == PLAYERS.O:
        pygame.draw.circle(screen, CIRCLE_COLOR, (center_x, center_y), CIRCLE_RADIUS, CIRCLE_WIDTH)
      if board[row][col] == PLAYERS.X:
        pygame.draw.line(screen, CROSS_COLOR, (center_x - 30, center_y - 30),
                                              (center_x + 30, center_y + 30), CROSS_WIDTH)  
        
        pygame.draw.line(screen, CROSS_COLOR, (center_x - 30, center_y + 30),
                                              (center_x + 30, center_y - 30), CROSS_WIDTH)

draw_lines()

games = Games(bs=1)
BATCH_SIZE = 1
credits = INIT_CREDS * torch.ones((BATCH_SIZE,), dtype=torch.int8, device=DEVICE)


good_weights = pickle.load(open('good_weights.pkl', 'rb'))
golden_params = {}
golden_params['input'] = torch.cat([good_weights[0].T[None,:,:].to(device=DEVICE)  for _ in range(BATCH_SIZE)], dim=0)
golden_params['bias'] = torch.cat([good_weights[1][None,:].to(device=DEVICE) for _ in range(BATCH_SIZE)], dim=0)
golden_params['output'] = torch.cat([good_weights[2].T[None,:,:].to(device=DEVICE) for _ in range(BATCH_SIZE)], dim=0)

players = Players(credits, golden_params)



while not games.game_over[0]:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False
    if event.type == pygame.MOUSEBUTTONDOWN:
      mouseX = event.pos[0]  # x
      mouseY = event.pos[1]  # y

      clicked_row = int(mouseY // 100)
      clicked_col = int(mouseX // 100)
      move = torch.zeros((3,3), device=DEVICE)
      
      move[clicked_row, clicked_col] = 1
      games.update(move.reshape((1,BOARD_SIZE)), PLAYERS.X, test=True)
      if not games.game_over[0]:
        move = players.play(games.boards, test=True)
        print(move)
        games.update(move.reshape((1,BOARD_SIZE)), PLAYERS.O, test=True)

  draw_figures(games.boards.cpu().reshape((3,3)).numpy())
  pygame.display.update()
  time.sleep(0.05)
if games.winners[0] == PLAYERS.X:
  print("X wins")
elif games.winners[0] == PLAYERS.O:
  print("O wins")
else:
  print("Nobody wins")
#time.sleep(5)

pygame.quit()
