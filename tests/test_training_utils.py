import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from training_utils import (
    apply_timeout_reward,
    decay_exploration_noise,
    episode_train_iterations,
    exploratory_action,
    replay_ready_for_updates,
    replay_done,
    single_ego_exploration_index,
)


class FakeRng:
    def uniform(self, low, high):
        return 0.25

    def normal(self, mean, scale, size):
        return np.full(size, 0.5, dtype=np.float32)


class TrainingUtilsTests(unittest.TestCase):
    def test_timeout_reward_only_replaces_unresolved_agents(self):
        rewards = apply_timeout_reward(
            [100.0, -100.0, 1.5], [True, True, False], True, -150.0
        )
        self.assertEqual(rewards, [100.0, -100.0, -150.0])

    def test_timeout_reward_preserves_non_truncated_step(self):
        rewards = apply_timeout_reward([1.0, 2.0], [False, False], False, -150.0)
        self.assertEqual(rewards, [1.0, 2.0])

    def test_timeout_reward_can_be_disabled(self):
        rewards = apply_timeout_reward([1.0], [False], True, None)
        self.assertEqual(rewards, [1.0])

    def test_timeout_reward_rejects_mismatched_inputs(self):
        with self.assertRaises(ValueError):
            apply_timeout_reward([1.0], [False, False], True, -150.0)

    def test_timeout_is_terminal_for_replay(self):
        self.assertEqual(replay_done(True, False), 1)
        self.assertEqual(replay_done(False, True), 1)
        self.assertEqual(replay_done(False, False), 0)

    def test_updates_match_joint_environment_steps(self):
        self.assertEqual(episode_train_iterations(100), 100)
        self.assertEqual(episode_train_iterations(1), 1)
        self.assertEqual(episode_train_iterations(0), 1)

    def test_update_scaling_rejects_negative_environment_steps(self):
        with self.assertRaises(ValueError):
            episode_train_iterations(-1)

    def test_replay_warmup_delays_all_updates(self):
        self.assertFalse(replay_ready_for_updates(0, 0))
        self.assertFalse(replay_ready_for_updates(4999, 5000))
        self.assertTrue(replay_ready_for_updates(5000, 5000))

    def test_replay_warmup_rejects_negative_minimum(self):
        with self.assertRaises(ValueError):
            replay_ready_for_updates(10, -1)

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

    def test_random_linear_exploration_preserves_policy_steering(self):
        action = exploratory_action(
            [0.9, -0.4], 0.1, 1.0, randomize_linear=True, rng=FakeRng()
        )
        np.testing.assert_allclose(action, [0.25, 0.1])

    def test_standard_exploration_noises_both_action_dimensions(self):
        action = exploratory_action(
            [0.2, -0.4], 0.1, 1.0, randomize_linear=False, rng=FakeRng()
        )
        np.testing.assert_allclose(action, [0.7, 0.1])

    def test_exploration_clips_actions(self):
        action = exploratory_action(
            [0.9, 0.9], 0.1, 1.0, randomize_linear=False, rng=FakeRng()
        )
        np.testing.assert_allclose(action, [1.0, 1.0])

    def test_single_ego_exploration_rotates_over_active_agents(self):
        active = [True, False, True, True]
        self.assertEqual(single_ego_exploration_index(active, 0), 0)
        self.assertEqual(single_ego_exploration_index(active, 1), 2)
        self.assertEqual(single_ego_exploration_index(active, 2), 3)
        self.assertEqual(single_ego_exploration_index(active, 3), 0)

    def test_single_ego_exploration_handles_no_active_agent(self):
        self.assertIsNone(single_ego_exploration_index([False, False], 0))

    def test_single_ego_exploration_rejects_negative_step(self):
        with self.assertRaises(ValueError):
            single_ego_exploration_index([True], -1)


if __name__ == "__main__":
    unittest.main()
