import numpy as np

from tinydreamer.game import PelletMaze
from tinydreamer.trajectory import collect_episode


def evaluate_policy(game, policy, amt_eps):
    total_rewards = 0.0
    total_pellets = 0
    mazes_cleared = 0

    for _ in range(amt_eps):
        episode = collect_episode(game, policy)
        game_reward = sum(step["reward"] for step in episode)
        pellets_eaten = len(game.starting_pellets) - len(game.pellets)

        total_rewards += game_reward
        total_pellets += pellets_eaten
        if game.won:
            mazes_cleared += 1

    return {
        "average_reward": total_rewards / amt_eps,
        "average_pellets": total_pellets / amt_eps,
        "mazes_cleared": mazes_cleared,
        "episodes": amt_eps,
    }


if __name__ == "__main__":
    game = PelletMaze()
    random_generator = np.random.default_rng(0)

    def random_policy(observation):
        return int(random_generator.integers(0, 4))

    results = evaluate_policy(game, random_policy, 100)
    print("Average reward:", round(results["average_reward"], 2))
    print("Average pellets eaten:", round(results["average_pellets"], 2))
    print("Mazes cleared:", results["mazes_cleared"], "out of", results["episodes"])
