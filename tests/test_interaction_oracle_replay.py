import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from interaction_oracle import interaction_mask, nearest_context_distance
from replay_buffer import ReplayBuffer


class InteractionOracleTests(unittest.TestCase):
    def test_mask_uses_nearest_valid_neighbor(self):
        no_neighbor = np.zeros(10, dtype=np.float32)
        near_neighbor = np.array(
            [0.0, 0.0, 2.5, 0.0, 1.0, 0.0, 0.0, 1.5, 0.0, 1.0],
            dtype=np.float32,
        )
        self.assertEqual(nearest_context_distance(no_neighbor), float("inf"))
        self.assertAlmostEqual(nearest_context_distance(near_neighbor), 1.5)
        self.assertEqual(
            interaction_mask([no_neighbor, near_neighbor], 2.0), [False, True]
        )

    def test_mask_accepts_batched_critic_context_slices(self):
        contexts = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.9, 0.0, 1.0],
                [0.0, 0.0, 2.1, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.assertEqual(interaction_mask(contexts, 2.0), [False, True, False])

    def test_invalid_context_and_threshold_are_rejected(self):
        with self.assertRaises(ValueError):
            nearest_context_distance(np.zeros(6), feature_dim=5)
        with self.assertRaises(ValueError):
            interaction_mask([np.zeros(5)], 0.0)


class InteractionReplayTests(unittest.TestCase):
    @staticmethod
    def add(buffer, value, interaction):
        state = np.array([value], dtype=np.float32)
        critic_state = np.array([value, value], dtype=np.float32)
        action = np.array([value, -value], dtype=np.float32)
        buffer.add_local_critic(
            state,
            critic_state,
            action,
            float(value),
            False,
            state + 1,
            critic_state + 1,
            interaction=interaction,
        )

    def test_interaction_sampling_and_eviction(self):
        buffer = ReplayBuffer(3, random_seed=1)
        self.add(buffer, 1, True)
        self.add(buffer, 2, False)
        self.add(buffer, 3, True)
        self.assertEqual(buffer.interaction_size(), 2)
        rewards = buffer.sample_local_critic_batch(10, interaction_only=True)[3]
        self.assertEqual(set(rewards.ravel().tolist()), {1.0, 3.0})

        self.add(buffer, 4, False)
        self.assertEqual(buffer.size(), 3)
        self.assertEqual(buffer.interaction_size(), 1)
        rewards = buffer.sample_local_critic_batch(10, interaction_only=True)[3]
        self.assertEqual(rewards.ravel().tolist(), [3.0])

    def test_checkpoint_round_trip_and_legacy_restore(self):
        buffer = ReplayBuffer(4, random_seed=1)
        self.add(buffer, 1, True)
        self.add(buffer, 2, False)
        restored = ReplayBuffer(1)
        restored.load_state_dict(buffer.state_dict())
        self.assertEqual(restored.size(), 2)
        self.assertEqual(restored.interaction_size(), 1)

        legacy_state = buffer.state_dict()
        legacy_state.pop("interaction_buffer")
        legacy_state["buffer"] = [item[:7] for item in legacy_state["buffer"]]
        legacy = ReplayBuffer(1)
        legacy.load_state_dict(legacy_state)
        self.assertEqual(legacy.size(), 2)
        self.assertEqual(legacy.interaction_size(), 0)
        self.assertIsNone(
            legacy.sample_local_critic_batch(4, interaction_only=True)
        )

    def test_mixed_sampling_prioritizes_interaction_experiences(self):
        buffer = ReplayBuffer(20, random_seed=1)
        for value in range(10):
            self.add(buffer, value, interaction=value < 8)
        rewards = buffer.sample_local_critic_batch(
            8, interaction_fraction=0.75
        )[3].ravel()
        interaction_draws = sum(value < 8 for value in rewards)
        self.assertEqual(len(set(rewards)), 8)
        self.assertGreaterEqual(interaction_draws, 6)

    def test_mixed_sampling_rejects_invalid_fraction(self):
        buffer = ReplayBuffer(4, random_seed=1)
        self.add(buffer, 1, True)
        with self.assertRaises(ValueError):
            buffer.sample_local_critic_batch(2, interaction_fraction=1.1)


if __name__ == "__main__":
    unittest.main()
