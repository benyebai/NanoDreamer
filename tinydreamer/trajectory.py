import numpy as np

from tinydreamer.game import PelletMaze


def collect_episode(game, policy):
    observation, _ = game.reset()
    episode = []

    while True:
        action = policy(observation)
        next_observation, reward, terminated, truncated, _ = game.step(action)

        step = {
            "observation": observation.copy(),
            "action": action,
            "reward": reward,
            "next_observation": next_observation.copy(),
            "terminated": terminated,
            "truncated": truncated,
        }
        episode.append(step)

        if terminated or truncated:
            break
        observation = next_observation

    return episode


if __name__ == "__main__":
    game = PelletMaze()
    random_generator = np.random.default_rng(0)

    def random_policy(observation):
        return int(random_generator.integers(0, 4))

    episode = collect_episode(game, random_policy)
    for number, step in enumerate(episode):
        print(number, step["action"], step["reward"])
    print("Total reward:", sum(step["reward"] for step in episode))
