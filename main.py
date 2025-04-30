# main.py

import gym
import slimevolleygym
import threading
import time

from agents.minimax_agent import MinimaxAgent
from agents.alphabeta_agent import SlimeVolleyAlphaBetaAgent
from utils.record_screen import record_screen


def run_slimevolley_minimax():
    agent = MinimaxAgent(depth=2)
    env = gym.make("SlimeVolley-v0")
    obs = env.reset()
    done = False
    total_reward = 0

    env.render()
    time.sleep(1)
    threading.Thread(target=record_screen, args=("videos/slimevolley_minimax.avi", 20)).start()

    while not done:
        action = agent.act(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()

    env.close()
    print(f"SlimeVolley Minimax Total Reward: {total_reward}")


def run_slimevolley_alphabeta():
    agent = SlimeVolleyAlphaBetaAgent(depth=2)
    env = gym.make("SlimeVolley-v0")
    obs = env.reset()
    done = False
    total_reward = 0

    env.render()
    time.sleep(1)
    threading.Thread(target=record_screen, args=("videos/slimevolley_alphabeta.avi", 20)).start()

    while not done:
        action = agent.act(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()

    env.close()
    print(f"SlimeVolley AlphaBeta Total Reward: {total_reward}")


if __name__ == "__main__":
    print("Running Minimax on SlimeVolley...")
    run_slimevolley_minimax()

    print("Running AlphaBeta on SlimeVolley...")
    run_slimevolley_alphabeta()
