import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from critic_calibration import (
    combine_actor_and_critic_context,
    discounted_n_step_target,
    infer_critic_state_dim,
    pairwise_order_counts,
    summarize_counterfactual_calibration,
)


class CriticCalibrationTests(unittest.TestCase):
    def test_critic_state_dim_is_read_from_checkpoint_weight(self):
        self.assertEqual(
            infer_critic_state_dim({"layer_1.weight": torch.zeros(800, 52)}), 52
        )

    def test_context_is_appended_without_changing_actor_state(self):
        actor_states = [np.arange(3, dtype=np.float32), np.ones(3)]
        critic_states = combine_actor_and_critic_context(
            actor_states, [np.array([4.0, 5.0]), np.array([6.0, 7.0])], 5
        )
        np.testing.assert_array_equal(critic_states[0], [0.0, 1.0, 2.0, 4.0, 5.0])
        np.testing.assert_array_equal(actor_states[0], [0.0, 1.0, 2.0])

    def test_context_width_must_match_critic_state_dim(self):
        with self.assertRaises(ValueError):
            combine_actor_and_critic_context([[1.0, 2.0]], [[3.0]], 4)

    def test_discounted_target_includes_bootstrap_after_rewards(self):
        target = discounted_n_step_target([1.0, 2.0], 0.5, 4.0)
        self.assertAlmostEqual(target, 3.0)

    def test_pairwise_order_counts_only_observed_non_ties(self):
        agreements, comparable = pairwise_order_counts(
            [3.0, 1.0, 2.0], [30.0, 10.0, 20.0]
        )
        self.assertEqual((agreements, comparable), (3, 3))

    def test_summary_compares_actions_within_same_state_only(self):
        records = [
            {
                "scenario_id": "s1",
                "anchor_step": 2,
                "ego_index": 0,
                "predicted_qmin": 2.0,
                "observed_n_step_target": 1.0,
                "repeatable": True,
            },
            {
                "scenario_id": "s1",
                "anchor_step": 2,
                "ego_index": 0,
                "predicted_qmin": 1.0,
                "observed_n_step_target": 2.0,
                "repeatable": True,
            },
            {
                "scenario_id": "s2",
                "anchor_step": 2,
                "ego_index": 0,
                "predicted_qmin": 100.0,
                "observed_n_step_target": 100.0,
                "repeatable": False,
            },
        ]
        summary = summarize_counterfactual_calibration(records)
        self.assertEqual(summary["state_action_groups"], 1)
        self.assertEqual(summary["pairwise_comparisons"], 1)
        self.assertEqual(summary["pairwise_order_accuracy"], 0.0)
        self.assertEqual(summary["repeatable_records"], 2)

    def test_invalid_discount_is_rejected(self):
        with self.assertRaises(ValueError):
            discounted_n_step_target([1.0], 1.1)


if __name__ == "__main__":
    unittest.main()
