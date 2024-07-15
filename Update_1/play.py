import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

pygame.init()

font = pygame.font.Font('arial.ttf', 25) # path arial

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN1 = (0, 255, 0)
GREEN2 = (0, 200, 0)
DARK_GRAY = (30, 30, 30)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)

COLORS = [GREEN1, YELLOW, BLUE]

BLOCK_SIZE = 20
SPEED = 60

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.apple_image = pygame.image.load('apple.jpg').convert_alpha() # path apple
        self.apple_image = pygame.transform.scale(self.apple_image, (BLOCK_SIZE, BLOCK_SIZE))
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.color_index = 0  # Khởi tạo màu sắc ban đầu

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        self._move(action)
        self.snake.insert(0, self.head)
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            self._save_score(self.score)
            return reward, game_over, self.score
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
            self.color_index = (self.color_index + 1) % len(COLORS)  
        else:
            self.snake.pop()
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill(DARK_GRAY)
        snake_color = COLORS[self.color_index]
        for pt in self.snake:
            pygame.draw.rect(self.display, snake_color, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, snake_color, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        self.display.blit(self.apple_image, (self.food.x, self.food.y))
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
        self.direction = new_dir
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)

    def _save_score(self, score):
        with open("scores_1.txt", "a") as f: # name save scores
            f.write(f"{score}\n")

def get_state(game):
    head = game.snake[0]
    point_l = Point(head.x - BLOCK_SIZE, head.y)
    point_r = Point(head.x + BLOCK_SIZE, head.y)
    point_u = Point(head.x, head.y - BLOCK_SIZE)
    point_d = Point(head.x, head.y + BLOCK_SIZE)
    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    state = [
        (dir_r and game.is_collision(point_r)) or
        (dir_l and game.is_collision(point_l)) or
        (dir_u and game.is_collision(point_u)) or
        (dir_d and game.is_collision(point_d)),
        (dir_u and game.is_collision(point_r)) or
        (dir_d and game.is_collision(point_l)) or
        (dir_l and game.is_collision(point_u)) or
        (dir_r and game.is_collision(point_d)),
        (dir_d and game.is_collision(point_r)) or
        (dir_u and game.is_collision(point_l)) or
        (dir_r and game.is_collision(point_u)) or
        (dir_l and game.is_collision(point_d)),

        dir_l,
        dir_r,
        dir_u,
        dir_d,

        game.food.x < game.head.x,
        game.food.x > game.head.x,
        game.food.y < game.head.y,
        game.food.y > game.head.y
    ]
    return np.array(state, dtype=int)

def main():
    model = DQN(11, 256, 3)
    model.load_state_dict(torch.load('model_1.pth')) # path model
    model.eval()
    game = SnakeGameAI()
    while True:
        state_old = get_state(game)
        state_old_tensor = torch.tensor(state_old, dtype=torch.float)
        with torch.no_grad():
            prediction = model(state_old_tensor)
        move = torch.argmax(prediction).item()
        action = [0, 0, 0]
        action[move] = 1
        reward, game_over, score = game.play_step(action)
        if game_over:
            game.reset()
        print(f"Score: {score}")

if __name__ == '__main__':
    main()
