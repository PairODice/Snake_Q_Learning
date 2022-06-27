from collections import deque
import numpy as np
from snake_game import Direction, Point, SnakeGame
from model import QTrainer, Linear_QNet
import torch
import random


def search(x, y, move_left, already_searched, board):
    if move_left == 0 or board._is_collision(Point(x,y)) or Point(x, y) in already_searched:
        return 0
    already_searched.append(Point(x, y))
    points = search(x - 20, y, move_left - 1, already_searched, board)
    points += search(x + 20, y, move_left - 1, already_searched, board)
    points += search(x, y - 20, move_left - 1, already_searched, board)
    points += search(x, y + 20, move_left - 1, already_searched, board)
    return 1 + points


def look_around(head, board, direction, num_moves):
    if direction == Direction.UP:
        moves_l = search(head.x - 20, head.y, num_moves, [], board)
        moves_r = search(head.x + 20, head.y, num_moves, [], board)
        moves_u = search(head.x, head.y - 20, num_moves, [], board)
    elif direction == Direction.DOWN:
        moves_l = search(head.x + 20, head.y, num_moves, [], board)
        moves_r = search(head.x - 20, head.y, num_moves, [], board)
        moves_u = search(head.x, head.y + 20, num_moves, [], board)
    elif direction == Direction.LEFT:
        moves_l = search(head.x, head.y + 20, num_moves, [], board)
        moves_r = search(head.x, head.y - 20, num_moves, [], board)
        moves_u = search(head.x - 20, head.y, num_moves, [], board)
    elif direction == Direction.RIGHT:
        moves_l = search(head.x, head.y - 20, num_moves, [], board)
        moves_r = search(head.x, head.y + 20, num_moves, [], board)
        moves_u = search(head.x + 20, head.y, num_moves, [], board)
    return moves_l, moves_r, moves_u


def danger_straight(direction, game):
    head = game.snake[0]
    if direction == Direction.RIGHT:
        return game._is_collision(Point(x=head.x + 20, y=head.y))
    if direction == Direction.LEFT:
        return game._is_collision(Point(x=head.x - 20, y=head.y))
    if direction == Direction.UP:
        return game._is_collision(Point(x=head.x, y=head.y - 20))
    if direction == Direction.DOWN:
        return game._is_collision(Point(x=head.x, y=head.y + 20))


def danger_left(direction, game):
    head = game.snake[0]
    if direction == Direction.RIGHT:
        return game._is_collision(Point(x=head.x, y=head.y - 20))
    if direction == Direction.LEFT:
        return game._is_collision(Point(x=head.x, y=head.y + 20))
    if direction == Direction.UP:
        return game._is_collision(Point(x=head.x - 20, y=head.y))
    if direction == Direction.DOWN:
        return game._is_collision(Point(x=head.x + 20, y=head.y))


def danger_right(direction, game):
    head = game.snake[0]
    if direction == Direction.RIGHT:
        return game._is_collision(Point(x=head.x, y=head.y + 20))
    if direction == Direction.LEFT:
        return game._is_collision(Point(x=head.x, y=head.y - 20))
    if direction == Direction.UP:
        return game._is_collision(Point(x=head.x + 20, y=head.y))
    if direction == Direction.DOWN:
        return game._is_collision(Point(x=head.x - 20, y=head.y))


def get_food_direction(game):
    head = game.snake[0]
    food = game.food

    if game.direction == Direction.UP:
        food_l = head.x > food.x
        food_r = head.x < food.x
        food_u = head.y > food.y
        food_d = head.y < food.y
        return food_l, food_r, food_u, food_d
    if game.direction == Direction.DOWN:
        food_l = head.x < food.x
        food_r = head.x > food.x
        food_u = head.y < food.y
        food_d = head.y > food.y
        return food_l, food_r, food_u, food_d
    if game.direction == Direction.LEFT:
        food_l = head.y > food.y
        food_r = head.x < food.x
        food_u = head.x > food.x
        food_d = head.x < food.x
        return food_l, food_r, food_u, food_d
    if game.direction == Direction.RIGHT:
        food_l = head.y < food.y
        food_r = head.x > food.x
        food_u = head.x < food.x
        food_d = head.x > food.x
        return food_l, food_r, food_u, food_d

BATCH_SIZE = 1000
LR = 0.001
class Agent:
    def __init__(self):

        self.gamma = 0                          # Discount rate
        self.memory = deque(maxlen=100_000)     # max memory
        self.model = Linear_QNet(10, 256, 3)    # Model Size
        self.trainer = QTrainer(self.model, LR, self.gamma) # Training parameters
        self.n_games = 0                        # number of games played
        self.epsilon = 100                      # randomness element, explore vs exploit
        self.n_moves = 5                        # how many moves to look ahead for

    def get_state(self, board):
        food_l, food_r, food_u, food_d = get_food_direction(board)
        self.n_moves = len(board.snake) + 2
        moves_l, moves_r, moves_u = look_around(board.snake[0], board, board.direction, self.n_moves)
        norm_val = 2 * self.n_moves ** 2 - 2 * self.n_moves + 1
        moves_l = moves_l / norm_val
        moves_r = moves_r / norm_val
        moves_u = moves_u / norm_val
        state = [
            # Danger straight
            danger_straight(board.direction, board),
            # Danger left
            danger_left(board.direction, board),
            # Danger right
            danger_right(board.direction, board),
            # current direction
            # board.direction == Direction.RIGHT,
            # board.direction == Direction.LEFT,
            # board.direction == Direction.UP,
            # board.direction == Direction.DOWN,
            # Food direction
            food_l,
            food_r,
            food_u,
            food_d,

            # Is it at a boarder
            # board.head.x == 0,
            # board.head.x == 620,
            # board.head.y == 0,
            # board.head.y == 460,

            # Number of available moves
            moves_l,
            moves_r,
            moves_u
        ]

        return np.array(state, dtype=float)

    def remember(self, action, old_state, reward, new_state, game_over):
        self.memory.append((action, old_state, reward, new_state, game_over))

    def make_decision(self, state):
        decision = [0, 0, 0]
        state = torch.tensor(state, dtype=torch.float)
        if random.randint(0, 200) < self.epsilon - self.n_games:
            decision[random.randint(0, 2)] = 1
        else:
            ai_decision = torch.argmax(self.model(state)).item()
            decision[ai_decision] = 1

        return decision

    def train_short_memory(self, action, old_state, reward, new_state, game_over):
        self.trainer.train_step(action, old_state, reward, new_state, game_over)

    def train_long_memory(self):
        if BATCH_SIZE < len(self.memory):
            mini_batch = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_batch = self.memory
        actions, old_states, rewards, new_states, game_overs = zip(*mini_batch)
        self.trainer.train_step(actions, old_states, rewards, new_states, game_overs)


if __name__ == '__main__':
    agent = Agent()
    game = SnakeGame()
    high_score = 0
    while True:
        old_state = agent.get_state(game)
        decision = agent.make_decision(old_state)

        game_over, score, fitness = game.play_step(decision)
        new_state = agent.get_state(game)
        agent.remember(decision, old_state, fitness, new_state, game_over)

        agent.train_short_memory(decision, old_state, fitness, new_state, game_over)

        if game_over:
            if score > 20 and score > high_score:
                agent.model.save(f'rewards_ckp_score{score}')
            if score > high_score:
                high_score = score
            print("Game:", agent.n_games, "Score:", score, "Record:", high_score)
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
