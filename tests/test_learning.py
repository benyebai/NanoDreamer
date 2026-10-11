import unittest

import numpy as np
from mininn.optimizers import Adam

from actor import Actor
from critic import Critic
from game import PelletMaze
from returns import calculate_returns, calculate_lambda_returns
from train_actor_critic import train_one_episode


class LearningTests(unittest.TestCase):
    def test_returns_with_and_without_bootstrap(self):
        episode = [{"reward": 1}, {"reward": 0}, {"reward": 2}]
        np.testing.assert_allclose(calculate_returns(episode, 0.9),
                                   [2.62, 1.8, 2.0])
        np.testing.assert_allclose(calculate_returns(episode, 0.9, 4.0),
                                   [5.536, 5.04, 5.6])

    def test_lambda_returns_by_hand(self):
        episode = [
            {"reward": 1, "terminated": False},
            {"reward": 0, "terminated": False},
            {"reward": 2, "terminated": True},
        ]
        next_values = [4, 5, 0]
        np.testing.assert_allclose(
            calculate_lambda_returns(episode, next_values, 0.9, 0.0),
            [4.6, 4.5, 2.0],
        )
        np.testing.assert_allclose(
            calculate_lambda_returns(episode, next_values, 0.9, 0.5),
            [4.2175, 3.15, 2.0],
        )
        np.testing.assert_allclose(
            calculate_lambda_returns(episode, next_values, 0.9, 1.0),
            calculate_returns(episode, 0.9),
        )

    def test_lambda_return_at_time_limit(self):
        episode = [
            {"reward": 1, "terminated": False},
            {"reward": 2, "terminated": False},
        ]
        np.testing.assert_allclose(
            calculate_lambda_returns(episode, [4, 5], 0.9, 0.5, 5),
            [5.725, 6.5],
        )

    def test_critic_learns_toward_one_target(self):
        game = PelletMaze()
        observation, _ = game.reset()
        critic = Critic(0)
        optimizer = Adam(critic.parameters(), learning_rate=0.001)
        before = critic.predict(observation)
        target = 10.0
        value = critic.forward(observation)[0, 0]
        loss = (value - target) * (value - target)
        critic.network.zero_grad()
        loss.backward()
        optimizer.step()
        after = critic.predict(observation)
        self.assertLess(abs(after - target), abs(before - target))

    def test_one_episode_updates_both_networks(self):
        actor = Actor(0)
        critic = Critic(1)
        actor_optimizer = Adam(actor.parameters(), learning_rate=0.001)
        critic_optimizer = Adam(critic.parameters(), learning_rate=0.001)
        actor_before = [p.data.copy() for p in actor.parameters()]
        critic_before = [p.data.copy() for p in critic.parameters()]

        reward, actor_loss, critic_loss = train_one_episode(
            actor, critic, PelletMaze(max_steps=5),
            actor_optimizer, critic_optimizer, 0.99
        )
        self.assertTrue(np.isfinite([reward, actor_loss, critic_loss]).all())
        self.assertTrue(any(np.any(p.data != old)
                            for p, old in zip(actor.parameters(), actor_before)))
        self.assertTrue(any(np.any(p.data != old)
                            for p, old in zip(critic.parameters(), critic_before)))


if __name__ == "__main__":
    unittest.main()
