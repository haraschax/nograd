#!/usr/bin/env python
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import argparse
import time
import torch
import pickle
from helpers import PLAYERS, BOARD_ROWS, BOARD_COLS, BOARD_SIZE
from batch_arena import Games, Players

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
  parser = argparse.ArgumentParser()
  parser.add_argument('--perfect', action='store_true', help='Play vs perfect player')
  args = parser.parse_args()

  draw_lines()
  device = 'cpu'
  games = Games(bs=1, device=device)

  if args.perfect:
    params = pickle.load(open('perfect_dna.pkl', 'rb'))
  else:
    params = pickle.load(open('organic_dna.pkl', 'rb'))
  players = Players.from_params(params, device=device)

  while not games.game_over[0]:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      if event.type == pygame.MOUSEBUTTONDOWN:
        mouseX = event.pos[0]
        mouseY = event.pos[1]

        clicked_row = int(mouseY // SQUARE_SIZE)
        clicked_col = int(mouseX // SQUARE_SIZE)
        move = torch.zeros((3,3), device=device)
        
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
