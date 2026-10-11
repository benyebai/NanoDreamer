import unittest

import numpy as np

from evaluate import evaluate_policy
from game import PelletMaze


class GhostTests(unittest.TestCase):
    def test_old_game_still_has_three_channels(self):
        game = PelletMaze()
        observation, _ = game.reset()
        self.assertEqual(observation.shape, (3, 3, 3))
        self.assertIsNone(game.ghost)

    def test_full_maze_channels(self):
        game = PelletMaze(with_ghost=True)
        game.reset()
        observation = game.observe()
        self.assertEqual(observation.shape, (7, 7, 4))
        self.assertEqual(observation[0, 0, 0], 1)  # wall
        self.assertEqual(observation[1, 2, 1], 1)  # pellet
        self.assertEqual(observation[1, 1, 2], 1)  # player
        self.assertEqual(observation[5, 5, 3], 1)  # ghost
        game.player = (5, 4)
        observation = game.observe()
        self.assertEqual(observation[5, 4, 2], 1)

    def test_old_ghost_actor_can_use_local_view(self):
        game = PelletMaze(with_ghost=True, full_view=False)
        observation, _ = game.reset()
        self.assertEqual(observation.shape, (3, 3, 4))
        game.player = (5, 4)
        observation = game.observe()
        self.assertEqual(observation[1, 2, 3], 1)

    def test_entering_ghost_ends_episode(self):
        game = PelletMaze(with_ghost=True)
        game.reset()
        game.player = (5, 4)
        _, reward, terminated, truncated, info = game.step(1)
        self.assertEqual(reward, -5.01)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info['caught'])
        self.assertFalse(info['won'])

    def test_ghost_moves_on_third_turn_and_can_catch_player(self):
        game = PelletMaze(with_ghost=True)
        game.reset()
        game.player = (5, 3)
        game.steps = 2
        _, _, terminated, _, info = game.step(1)
        self.assertTrue(terminated)
        self.assertTrue(info['caught'])
        self.assertEqual(game.ghost, game.player)

    def test_snapshot_restores_ghost(self):
        game = PelletMaze(with_ghost=True)
        game.reset()
        saved = game.snapshot()
        game.ghost = (4, 5)
        game.restore(saved)
        self.assertEqual(game.ghost, (5, 5))

    def test_caught_episode_is_not_a_maze_clear(self):
        game = PelletMaze(with_ghost=True)
        def into_ghost(_):
            game.player = (5, 4)
            return 1
        result = evaluate_policy(game, into_ghost, 1)
        self.assertEqual(result['mazes_cleared'], 0)
        self.assertTrue(game.caught)


if __name__ == '__main__':
    unittest.main()
