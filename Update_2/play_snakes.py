import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

pygame.init()
font = pygame.font.SysFont('arial.ttf', 25) # path arial

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
SNAKE_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
FOOD_IMAGES = ['apple.jpg',  # path apple
               'banana.jpg', # path banana
               'kiwi.jpg',   # path kiwi
               'peach.jpg']  # path peach

BLOCK_SIZE = 20
SPEED = 10

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

    def __init__(self, w=640, h=480, num_snakes=5):
        self.w = w
        self.h = h
        self.num_snakes = num_snakes
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
        self.load_food_images()

    def load_food_images(self):
        self.food_images = [pygame.image.load(img).convert_alpha() for img in FOOD_IMAGES]
        for i, img in enumerate(self.food_images):
            self.food_images[i] = pygame.transform.scale(img, (BLOCK_SIZE, BLOCK_SIZE))

    def reset(self, snake=None):
        if snake is None:
            self.snakes = []
            for i in range(self.num_snakes):
                self.snakes.append({
                    'direction': random.choice(list(Direction)),
                    'head': Point(random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE,
                                  random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE),
                    'body': [],
                    'score': 0,
                    'color': SNAKE_COLORS[i],
                    'food': None,
                    'frame_iteration': 0
                })
                self.snakes[-1]['body'] = [self.snakes[-1]['head'],
                                           Point(self.snakes[-1]['head'].x - BLOCK_SIZE, self.snakes[-1]['head'].y),
                                           Point(self.snakes[-1]['head'].x - (2 * BLOCK_SIZE), self.snakes[-1]['head'].y)]
                self._place_food(self.snakes[-1])
        else:
            snake['direction'] = random.choice(list(Direction))
            snake['head'] = Point(random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE,
                                  random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE)
            snake['body'] = [snake['head'],
                             Point(snake['head'].x - BLOCK_SIZE, snake['head'].y),
                             Point(snake['head'].x - (2 * BLOCK_SIZE), snake['head'].y)]
            snake['score'] = 0
            snake['frame_iteration'] = 0
            self._place_food(snake)

    def _place_food(self, snake):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        snake['food'] = Point(x, y)
        if snake['food'] in snake['body']:
            self._place_food(snake)

    def play_step(self, actions):
        for idx, snake in enumerate(self.snakes):
            snake['frame_iteration'] += 1

            # Move snake
            self._move(actions[idx], snake)
            snake['body'].insert(0, snake['head'])

            # Check if game over
            reward = 0
            game_over = False
            if self.is_collision(snake) or snake['frame_iteration'] > 100 * len(snake['body']):
                game_over = True
                reward = -10
                self.reset(snake)
                continue

            # Place new food or just move
            if snake['head'] == snake['food']:
                snake['score'] += 1
                reward = 10
                self._place_food(snake)
            else:
                snake['body'].pop()

            # Check for collisions between snakes
            for other_snake in self.snakes:
                if other_snake != snake and self.is_collision(snake, other_snake['head']):
                    self.reset(snake)
                    self.reset(other_snake)

        # Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, False, [snake['score'] for snake in self.snakes]

    def is_collision(self, snake, pt=None):
        if pt is None:
            pt = snake['head']
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in snake['body'][1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)
        for idx, snake in enumerate(self.snakes):
            for pt in snake['body']:
                pygame.draw.rect(self.display, snake['color'], pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            self.display.blit(random.choice(self.food_images), (snake['food'].x, snake['food'].y))
        for i, snake in enumerate(self.snakes):
            score_text = f"Snake {i+1}: {snake['score']}"
            text = font.render(score_text, True, snake['color'])
            self.display.blit(text, [10, 30 * i])
        pygame.display.flip()

    def _move(self, action, snake):
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(snake['direction'])
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d
        snake['direction'] = new_dir
        x = snake['head'].x
        y = snake['head'].y
        if snake['direction'] == Direction.RIGHT:
            x += BLOCK_SIZE
        elif snake['direction'] == Direction.LEFT:
            x -= BLOCK_SIZE
        elif snake['direction'] == Direction.DOWN:
            y += BLOCK_SIZE
        elif snake['direction'] == Direction.UP:
            y -= BLOCK_SIZE
        snake['head'] = Point(x, y)

    def show_game_over(self):
        self.display.fill(BLACK)
        text1 = font.render("Game Over", True, RED)
        text2 = font.render("Press any key to play again", True, WHITE)
        self.display.blit(text1, [self.w // 2 - text1.get_width() // 2, self.h // 2 - text1.get_height() // 2])
        self.display.blit(text2, [self.w // 2 - text2.get_width() // 2, self.h // 2 + text1.get_height() // 2])
        pygame.display.flip()
        self.wait_for_key_press()

    def wait_for_key_press(self):
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    waiting = False

def get_state(game, snake):
    head = snake['body'][0]
    point_l = Point(head.x - BLOCK_SIZE, head.y)
    point_r = Point(head.x + BLOCK_SIZE, head.y)
    point_u = Point(head.x, head.y - BLOCK_SIZE)
    point_d = Point(head.x, head.y + BLOCK_SIZE)
    
    dir_l = snake['direction'] == Direction.LEFT
    dir_r = snake['direction'] == Direction.RIGHT
    dir_u = snake['direction'] == Direction.UP
    dir_d = snake['direction'] == Direction.DOWN

    state = [
        # Danger straight
        (dir_r and game.is_collision(snake, point_r)) or 
        (dir_l and game.is_collision(snake, point_l)) or 
        (dir_u and game.is_collision(snake, point_u)) or 
        (dir_d and game.is_collision(snake, point_d)),

        # Danger right
        (dir_u and game.is_collision(snake, point_r)) or 
        (dir_d and game.is_collision(snake, point_l)) or 
        (dir_l and game.is_collision(snake, point_u)) or 
        (dir_r and game.is_collision(snake, point_d)),

        # Danger left
        (dir_d and game.is_collision(snake, point_r)) or 
        (dir_u and game.is_collision(snake, point_l)) or 
        (dir_r and game.is_collision(snake, point_u)) or 
        (dir_l and game.is_collision(snake, point_d)),

        # Move direction
        dir_l,
        dir_r,
        dir_u,
        dir_d,

        # Food location 
        snake['food'].x < head.x,  # food left
        snake['food'].x > head.x,  # food right
        snake['food'].y < head.y,  # food up
        snake['food'].y > head.y  # food down
    ]

    return np.array(state, dtype=int)

def main():
    # Load the trained model
    model = DQN(11, 256, 3)
    model.load_state_dict(torch.load('model_DQLN.pth')) # path model
    model.eval()

    game = SnakeGameAI()

    while True:
        states_old = [get_state(game, snake) for snake in game.snakes]
        states_old_tensors = [torch.tensor(state, dtype=torch.float) for state in states_old]

        # Predict actions based on the model
        actions = []
        for state in states_old_tensors:
            with torch.no_grad():
                prediction = model(state)
            move = torch.argmax(prediction).item()
            action = [0, 0, 0]
            action[move] = 1
            actions.append(action)

        reward, game_over, scores = game.play_step(actions)
        if game_over:
            game.show_game_over()
            game.reset()

        print(f"Scores: {scores}")

if __name__ == '__main__':
    main()
