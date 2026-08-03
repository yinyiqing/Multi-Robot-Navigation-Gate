import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from critic_calibration import (
    calibration_reward_kwargs,
    combine_actor_and_critic_context,
    discounted_n_step_target,
    infer_critic_state_dim,
    manifest_conflict_pair_indices,
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

    def test_dense_v9_reward_profile_matches_training_contract(self):
        profile = calibration_reward_kwargs("dense_v9")
        self.assertEqual(profile["progress_reward_weight"], 20.0)
        self.assertEqual(profile["robot_safe_distance"], 1.2)
        self.assertEqual(profile["robot_proximity_speed_penalty_weight"], 5.0)
        self.assertTrue(profile["safe_recovery_reward"])

    def test_individual_simple_profile_has_no_reward_coupling(self):
        profile = calibration_reward_kwargs("individual_simple")
        self.assertFalse(profile["cooperative_reward"])
        self.assertEqual(profile["progress_reward_weight"], 10.0)
        self.assertEqual(profile["robot_safe_distance"], 0.0)

    def test_manifest_conflict_pair_is_resolved_by_agent_name(self):
        case = {
            "metrics": {"conflict_edges": [{"agents": ["r4", "r2"]}]}
        }
        self.assertEqual(
            manifest_conflict_pair_indices(case, ["r1", "r2", "r3", "r4"]),
            [3, 1],
        )

    def test_manifest_conflict_pair_rejects_multiple_edges(self):
        case = {
            "metrics": {
                "conflict_edges": [
                    {"agents": ["r1", "r2"]},
                    {"agents": ["r2", "r3"]},
                ]
            }
        }
        with self.assertRaisesRegex(ValueError, "exactly one edge"):
            manifest_conflict_pair_indices(case, ["r1", "r2", "r3"])


if __name__ == "__main__":
    unittest.main()
