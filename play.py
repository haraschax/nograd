#!/usr/bin/env python
import argparse
import os
import time

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import pygame

from helpers import BOARD_COLS, BOARD_ROWS, BOARD_SIZE, PLAYERS
from batch_arena import Games, is_draw, is_winner, load_players, perfect_move


SQUARE_SIZE = 100
WIDTH, HEIGHT = BOARD_ROWS * SQUARE_SIZE, BOARD_COLS * SQUARE_SIZE
LINE_WIDTH = 10
MARK_SIZE = 70
CROSS_SIZE = 30
CIRCLE_RADIUS = MARK_SIZE // 2
CIRCLE_WIDTH = 10
CROSS_WIDTH = 25

BG_COLOR = (28, 170, 156)
LINE_COLOR = (23, 145, 135)
CIRCLE_COLOR = (239, 231, 200)
CROSS_COLOR = (66, 66, 66)


def draw_lines(screen):
  for i in range(1, BOARD_ROWS):
    pygame.draw.line(screen, LINE_COLOR, (0, i * SQUARE_SIZE), (WIDTH, i * SQUARE_SIZE), LINE_WIDTH)
  for i in range(1, BOARD_COLS):
    pygame.draw.line(screen, LINE_COLOR, (i * SQUARE_SIZE, 0), (i * SQUARE_SIZE, HEIGHT), LINE_WIDTH)


def draw_figures(screen, board):
  for row in range(BOARD_ROWS):
    for col in range(BOARD_COLS):
      center_x = int(col * SQUARE_SIZE + SQUARE_SIZE / 2)
      center_y = int(row * SQUARE_SIZE + SQUARE_SIZE / 2)
      value = board[row * BOARD_COLS + col]
      if value == PLAYERS.O:
        pygame.draw.circle(screen, CIRCLE_COLOR, (center_x, center_y), CIRCLE_RADIUS, CIRCLE_WIDTH)
      if value == PLAYERS.X:
        pygame.draw.line(
          screen,
          CROSS_COLOR,
          (center_x - CROSS_SIZE, center_y - CROSS_SIZE),
          (center_x + CROSS_SIZE, center_y + CROSS_SIZE),
          CROSS_WIDTH,
        )
        pygame.draw.line(
          screen,
          CROSS_COLOR,
          (center_x - CROSS_SIZE, center_y + CROSS_SIZE),
          (center_x + CROSS_SIZE, center_y - CROSS_SIZE),
          CROSS_WIDTH,
        )


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--perfect', action='store_true', help='Play against a minimax-perfect player')
  parser.add_argument('--dna-path', default='organic_dna.pkl')
  return parser.parse_args()


if __name__ == '__main__':
  args = parse_args()
  pygame.init()
  screen = pygame.display.set_mode((WIDTH, HEIGHT))
  pygame.display.set_caption('Tic Tac Toe')
  screen.fill(BG_COLOR)
  draw_lines(screen)

  games = Games(bs=1)
  players = None if args.perfect else load_players(args.dna_path)
  running = True

  while running and not games.game_over[0]:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      if event.type == pygame.MOUSEBUTTONDOWN:
        mouse_x, mouse_y = event.pos
        clicked_row = int(mouse_y // SQUARE_SIZE)
        clicked_col = int(mouse_x // SQUARE_SIZE)
        move = clicked_row * BOARD_COLS + clicked_col
        human_move = [-1e9] * BOARD_SIZE
        human_move[move] = 1.0
        games.update(human_move, PLAYERS.X, test=True)
        if not games.game_over[0]:
          if args.perfect:
            ai_move = perfect_move(games.boards[0], PLAYERS.O)
          else:
            ai_move = players.choose_move(games.boards[0], PLAYERS.O)
          ai_scores = [-1e9] * BOARD_SIZE
          ai_scores[ai_move] = 1.0
          games.update(ai_scores, PLAYERS.O, test=True)

    screen.fill(BG_COLOR)
    draw_lines(screen)
    draw_figures(screen, games.boards[0])
    pygame.display.update()
    time.sleep(0.05)

  if games.winners[0] == PLAYERS.X:
    print('X wins')
  elif games.winners[0] == PLAYERS.O:
    print('O wins')
  elif is_draw(games.boards[0]) or not running:
    print('Nobody wins')
  elif is_winner(games.boards[0], PLAYERS.X):
    print('X wins')
  else:
    print('Nobody wins')
  time.sleep(1)
  pygame.quit()
