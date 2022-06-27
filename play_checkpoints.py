from collections import deque
import numpy as np
from snake_game import Direction, Point, SnakeGame
from model import QTrainer, Linear_QNet
import torch
import random
from agent import Agent

agent = Agent()
agent.n_games = 100
agent.model.load_state_dict(torch.load('./checkpoints/ckp_score80'))
game = SnakeGame()
# TALK ABOUT HOW MODEL DIFFERENT AT START AND AT END OF GAME
while True:
    old_state = agent.get_state(game)
    decision = agent.make_decision(old_state)

    game_over, score, fitness = game.play_step(decision)
    new_state = agent.get_state(game)
    # agent.remember(decision, old_state, fitness, new_state, game_over)

    agent.train_short_memory(decision, old_state, fitness, new_state, game_over)
    if game_over:
        break
