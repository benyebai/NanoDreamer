import numpy as np

from game import PelletMaze
from trajectory import collect_episode


def calculate_returns(episode, discount, bootstrap_value=0.0):
    returns = []
    cumulative_reward = bootstrap_value

    for step in reversed(episode):
        cumulative_reward = step["reward"] + discount * cumulative_reward
        returns.append(cumulative_reward)

    returns.reverse()
    return returns


def calculate_lambda_returns(episode, next_values, discount, lam, bootstrap_value=0.0):
    returns = []
    running_return = bootstrap_value

    for step, next_value in zip(reversed(episode), reversed(next_values)):
        if step["terminated"]:
            running_return = step["reward"]
        else:
            guess = (1 - lam) * next_value + lam * running_return
            running_return = step["reward"] + discount * guess
        returns.append(running_return)

    returns.reverse()
    return returns


if __name__ == "__main__":
    game = PelletMaze()
    random_generator = np.random.default_rng(0)

    def random_policy(observation):
        return int(random_generator.integers(0, 4))

    episode = collect_episode(game, random_policy)
    returns = calculate_returns(episode, 0.9)

    for step, value in zip(episode, returns):
        print("reward:", step["reward"], "return:", value)
