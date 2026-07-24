import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_interaction_replay import has_local_critic_replay, interaction_band_masks


class InteractionReplayAuditTests(unittest.TestCase):
    def test_accepts_legacy_and_interaction_labeled_replay(self):
        self.assertTrue(has_local_critic_replay([tuple(range(7))]))
        self.assertTrue(has_local_critic_replay([tuple(range(8))]))
        self.assertFalse(has_local_critic_replay([]))
        self.assertFalse(has_local_critic_replay([tuple(range(5))]))

    def test_distance_bands_are_disjoint_and_complete(self):
        state_dim = 2
        feature_dim = 5
        critic_states = np.zeros((5, state_dim + 2 * feature_dim), dtype=np.float32)
        for index, distance in enumerate((3.0, 1.5, 1.0, 0.6), start=1):
            critic_states[index, state_dim + 2] = distance
            critic_states[index, state_dim + 4] = 1.0

        nearest, counts, masks = interaction_band_masks(
            critic_states, state_dim, feature_dim
        )

        self.assertTrue(np.isinf(nearest[0]))
        self.assertEqual(counts.tolist(), [0, 1, 1, 1, 1])
        self.assertEqual(
            {name: np.flatnonzero(mask).tolist() for name, mask in masks.items()},
            {
                "no_visible_neighbor": [0],
                "critical_le_0p8": [4],
                "near_0p8_to_1p2": [3],
                "interaction_1p2_to_2p0": [2],
                "visible_far_gt_2p0": [1],
            },
        )
        self.assertTrue(np.all(sum(mask.astype(int) for mask in masks.values()) == 1))


if __name__ == "__main__":
    unittest.main()
