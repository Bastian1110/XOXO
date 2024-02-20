"""
Script to train a XOXO agent
By : Sebastian Mora (@bastian1110)
"""
from stable_baselines3 import DQN
from TicTacToe import TicTacToe
from time import time


def create_dqn_agent():
    print("Training new Connect 4 DQN agent")
    print("Loading environment")
    env = TicTacToe()

    print("Training agent with Single Agent Self Play")
    start = time()
    model = DQN(
        "MlpPolicy",
        env,
        verbose=2,
        learning_rate=0.0005,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
    )
    model.learn(total_timesteps=10_000_000)

    end = time()

    model.save("./models/XOXOv1.0")
    print(f"Agent trained in {(end - start) / 60} minutes")


create_dqn_agent()
