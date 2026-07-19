import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from training_utils import (
    decay_exploration_noise,
    episode_train_iterations,
    replay_done,
)


class TrainingUtilsTests(unittest.TestCase):
    def test_timeout_is_terminal_for_replay(self):
        self.assertEqual(replay_done(True, False), 1)
        self.assertEqual(replay_done(False, True), 1)
        self.assertEqual(replay_done(False, False), 0)

    def test_updates_scale_with_collective_steps(self):
        self.assertEqual(episode_train_iterations(500, 5), 100)
        self.assertEqual(episode_train_iterations(501, 5), 101)
        self.assertEqual(episode_train_iterations(0, 5), 1)

    def test_update_scaling_rejects_invalid_agent_count(self):
        with self.assertRaises(ValueError):
            episode_train_iterations(10, 0)

    def test_exploration_decay_uses_configured_initial_value(self):
        value = 0.05
        for _ in range(80_000):
            value = decay_exploration_noise(value, 0.05, 0.02, 80_000)
        self.assertAlmostEqual(value, 0.02)

    def test_exploration_decay_clamps_at_minimum(self):
        self.assertEqual(decay_exploration_noise(0.02, 0.05, 0.02, 10), 0.02)

    def test_exploration_decay_rejects_invalid_configuration(self):
        with self.assertRaises(ValueError):
            decay_exploration_noise(0.05, 0.05, 0.02, 0)
        with self.assertRaises(ValueError):
            decay_exploration_noise(0.01, 0.01, 0.02, 10)


if __name__ == "__main__":
    unittest.main()
