"""Check run protection and evaluation."""

import tempfile
import unittest
from pathlib import Path

import torch

from nanodreamer.agent import Agent
from nanodreamer.config import Config
from scripts.train import evaluate, run


class TrainingTests(unittest.TestCase):
    def test_fresh_run_refuses_existing_results(self):
        with tempfile.TemporaryDirectory() as folder:
            result = Path(folder) / "result.json"
            result.write_text("original evidence")
            with self.assertRaises(FileExistsError):
                run("breakout", folder, steps=1)
            self.assertEqual(result.read_text(), "original evidence")

    def test_resume_refuses_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                run("breakout", folder, steps=1, resume=True)
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_evaluation_respects_cap_and_preserves_training_rng(self):
        torch.set_num_threads(2)
        agent = Agent(4, 3, config=Config(width=32, layers=1, context=4))
        random_state = torch.get_rng_state().clone()
        result = evaluate(agent, "breakout", episodes=2, cap=3, seed=7)
        self.assertTrue(torch.equal(torch.get_rng_state(), random_state))
        for mode in ("random", "agent"):
            self.assertEqual(len(result[mode]["scores"]), 2)
            self.assertTrue(all(0 < length <= 3 for length in result[mode]["lengths"]))


if __name__ == "__main__":
    unittest.main()
