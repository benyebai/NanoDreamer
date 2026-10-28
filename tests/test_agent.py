import tempfile
import unittest

import numpy as np
import torch

from nanodreamer.agent import Agent
from nanodreamer.config import Config
from nanodreamer.losses import lambda_returns, twohot
from nanodreamer.models import WorldModel
from nanodreamer.replay import Replay


class TransformerAgentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(1)
        self.config = Config(
            width=32, layers=1, context=4, batch=2, imagination_batch=2
        )

    def test_future_observations_do_not_change_earlier_states(self):
        model = WorldModel(4, 3, self.config).eval()
        z = torch.randn(2, 4, 64)
        actions = torch.zeros(2, 4, dtype=torch.long)
        first = model.states(z, actions)
        changed = z.clone()
        changed[:, 3] += 100
        second = model.states(changed, actions)
        torch.testing.assert_close(first[:, :3], second[:, :3])

    def test_terminal_reward_survives_termination_mask(self):
        r = torch.tensor([[1.0, 2.0]])
        con = torch.tensor([[1.0, 0.0]])
        values = torch.tensor([[100.0, 999.0]])
        result = lambda_returns(r, con, values, discount=1.0, lam=1.0)
        torch.testing.assert_close(result, torch.tensor([[3.0, 2.0]]))

    def test_replay_boundary_bootstraps_last_available_next_value(self):
        result = lambda_returns(
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[10.0, 100.0, 1000.0]]),
            discount=1.0,
            lam=1.0,
            valid=torch.tensor([[1.0, 0.0, 0.0]]),
        )
        self.assertEqual(result[0, 0].item(), 11.0)

    def test_short_episodes_are_sampled_without_crossing_resets(self):
        replay = Replay(self.config, 4, 3, 1)
        frame = np.zeros((4, 10, 10), bool)
        for marker in (0, 1):
            replay.start(frame + marker)
            replay.add(1, marker, 0.0, frame + marker)
        data = replay.sample(100)
        self.assertTrue((data["valid"].sum(-1) == 1).all())
        self.assertTrue((data["future_valid"].sum(-1) == 1).all())
        np.testing.assert_array_equal(data["frames"][:, -1], data["future"][:, 0])

    def test_twohot_mass_and_zero_value(self):
        labels = twohot(torch.tensor([-1e6, -1.0, 0.0, 1.0, 1e6]))
        torch.testing.assert_close(labels.sum(-1), torch.ones(5))
        self.assertEqual(labels[2, 127].item(), 1.0)

    def test_complete_update_has_finite_losses_and_no_actor_gradient_into_world(self):
        replay = Replay(self.config, 4, 3, 1)
        rng = np.random.default_rng(7)
        for _ in range(4):
            replay.start(rng.random((4, 10, 10)) < 0.05)
            for t in range(7):
                replay.add(
                    t % 3, float(t == 3), float(t != 6), rng.random((4, 10, 10)) < 0.05
                )
        agent = Agent(4, 3, config=self.config)
        result = agent.update(replay)
        self.assertTrue(all(np.isfinite(v) for v in result.values()))
        self.assertTrue(all(p.grad is None for p in agent.world.parameters()))
        self.assertTrue(any(p.grad is not None for p in agent.actor.parameters()))
        self.assertTrue(any(p.grad is not None for p in agent.critic.parameters()))

    def test_full_resume_reproduces_the_next_training_update(self):
        from minatar import Environment

        from scripts.train import load, save

        replay = Replay(self.config, 4, 3, 5)
        rng = np.random.default_rng(5)
        for _ in range(3):
            replay.start(rng.random((4, 10, 10)) < 0.05)
            for t in range(6):
                replay.add(
                    t % 3, float(t == 2), float(t != 5), rng.random((4, 10, 10)) < 0.05
                )
        agent = Agent(4, 3, config=self.config)
        with tempfile.TemporaryDirectory() as folder:
            save(folder, agent, replay, Environment("breakout"), {"step": 1}, rng)
            expected = agent.update(replay)
            restored, restored_replay, _, _, _ = load(folder, "cpu")
            actual = restored.update(restored_replay)
            for key in expected:
                self.assertAlmostEqual(expected[key], actual[key], places=5)
            for name in ("world", "actor", "critic"):
                for a, b in zip(
                    getattr(agent, name).parameters(),
                    getattr(restored, name).parameters(),
                ):
                    torch.testing.assert_close(a, b)


if __name__ == "__main__":
    unittest.main()
