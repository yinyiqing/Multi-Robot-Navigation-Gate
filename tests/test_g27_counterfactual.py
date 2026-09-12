import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from g27_counterfactual import (
    AGENT_NAMES,
    action_stratum,
    alignment_metrics,
    analyze_pilot,
    base_reward_components,
    distance_stratum,
    select_stratified_anchors,
    select_stratified_anchors_matching,
    selection_hash,
    validate_branch_spec,
)


def snapshot(position_offset=0.0, active_mask=None):
    return {
        "agent_names": list(AGENT_NAMES),
        "active_mask": active_mask or [True] * 5,
        "agents": {
            name: {
                "position": [index + position_offset, 0.0],
                "yaw": 0.0,
                "linear_velocity": [0.0, 0.0],
                "angular_velocity": 0.0,
                "actor_state": [0.0] * 24,
            }
            for index, name in enumerate(AGENT_NAMES)
        },
    }


def branch(reward, passed=True, target=False, collision=False):
    return {
        "reward": reward,
        "target": target,
        "collision": collision,
        "alignment": {"passed": passed},
    }


class RewardTest(unittest.TestCase):
    def test_nonterminal_reward_is_decomposed_exactly(self):
        result = base_reward_components(False, False, [0.4, -0.5], 0.7, 0.02)
        self.assertAlmostEqual(result["progress"], 0.4)
        self.assertAlmostEqual(result["forward"], 0.2)
        self.assertAlmostEqual(result["turn"], -0.1)
        self.assertAlmostEqual(result["obstacle"], -0.15)
        self.assertAlmostEqual(result["total"], 0.35)

    def test_terminal_reward_has_no_shaping_terms(self):
        goal = base_reward_components(True, False, [1.0, 1.0], 0.1, 1.0)
        collision = base_reward_components(False, True, [1.0, 1.0], 2.0, 1.0)
        self.assertEqual(goal["total"], 100.0)
        self.assertEqual(collision["total"], -100.0)
        self.assertEqual(goal["progress"], 0.0)
        self.assertEqual(collision["forward"], 0.0)


class SelectionTest(unittest.TestCase):
    def test_boundaries_and_hash_are_deterministic(self):
        self.assertEqual(distance_stratum(1.2), "deep")
        self.assertEqual(distance_stratum(2.0), "interaction")
        self.assertEqual(distance_stratum(3.0), "near")
        self.assertEqual(action_stratum(0.249), "low")
        self.assertEqual(action_stratum(0.25), "high")
        self.assertEqual(selection_hash(7, "case", 4, 1), selection_hash(7, "case", 4, 1))

    def test_selection_fills_each_cell_without_reusing_scenes(self):
        candidates = []
        for distance in ("deep", "interaction", "near", "far"):
            for action in ("low", "high"):
                for index in range(2):
                    scenario = "%s-%s-%d" % (distance, action, index)
                    candidates.append(
                        {
                            "scenario_id": scenario,
                            "distance_stratum": distance,
                            "action_stratum": action,
                            "selection_hash": selection_hash(1, scenario, 4, 0),
                        }
                    )
        selected = select_stratified_anchors(candidates, per_cell=2)
        self.assertEqual(len(selected), 16)
        self.assertEqual(len({item["scenario_id"] for item in selected}), 16)

    def test_matching_recovers_from_greedy_scene_conflicts(self):
        candidates = []
        cells = [
            (distance, action)
            for distance in ("deep", "interaction", "near", "far")
            for action in ("low", "high")
        ]
        for cell_index, (distance, action) in enumerate(cells):
            dedicated = "dedicated-%d" % cell_index
            shared = "shared-%d" % cell_index
            for rank, scenario in enumerate((shared, dedicated)):
                candidates.append(
                    {
                        "scenario_id": scenario,
                        "distance_stratum": distance,
                        "action_stratum": action,
                        "selection_hash": "%02d-%d" % (rank, cell_index),
                    }
                )
            if cell_index + 1 < len(cells):
                next_distance, next_action = cells[cell_index + 1]
                candidates.append(
                    {
                        "scenario_id": shared,
                        "distance_stratum": next_distance,
                        "action_stratum": next_action,
                        "selection_hash": "00-shared-%d" % cell_index,
                    }
                )
        selected = select_stratified_anchors_matching(candidates, per_cell=1)
        self.assertEqual(len(selected), 8)
        self.assertEqual(len({item["scenario_id"] for item in selected}), 8)


class SchemaAndAlignmentTest(unittest.TestCase):
    def test_branch_schema_rejects_wrong_prefix_length(self):
        specification = {
            "format_version": 1,
            "scenario_id": "case",
            "scenario_index": 0,
            "anchor_step": 2,
            "ego_index": 0,
            "agent_names": list(AGENT_NAMES),
            "action_prefix": [[[0.0, 0.0]] * 5],
            "anchor_snapshot": snapshot(),
            "navigation_actions": [[0.0, 0.0]] * 5,
            "interaction_actions": [[0.0, 0.0]] * 5,
        }
        with self.assertRaisesRegex(ValueError, "prefix length"):
            validate_branch_spec(specification)

    def test_alignment_uses_frozen_thresholds_and_active_mask(self):
        passed = alignment_metrics(snapshot(), snapshot(position_offset=0.019))
        failed = alignment_metrics(snapshot(), snapshot(position_offset=0.021))
        mask_failed = alignment_metrics(
            snapshot(), snapshot(active_mask=[True, True, True, True, False])
        )
        self.assertTrue(passed["passed"])
        self.assertFalse(failed["passed"])
        self.assertFalse(mask_failed["passed"])


class PilotCriteriaTest(unittest.TestCase):
    def test_pilot_pass_requires_both_actor_directions(self):
        records = []
        for index in range(8):
            if index < 4:
                rewards = (0.00, 0.01, 1.00, 1.01)
            else:
                rewards = (1.00, 1.01, 0.00, 0.01)
            records.append(
                {
                    "anchor_id": str(index),
                    "branches": {
                        name: branch(value)
                        for name, value in zip(("N1", "N2", "I1", "I2"), rewards)
                    },
                }
            )
        result = analyze_pilot(records)
        self.assertTrue(result["passed"])
        self.assertEqual(result["resolved_actor_support"]["navigation"], 4)
        self.assertEqual(result["resolved_actor_support"]["interaction"], 4)

    def test_alignment_failure_trips_pilot(self):
        records = []
        for index in range(10):
            branches = {
                "N1": branch(0.0),
                "N2": branch(0.0),
                "I1": branch(1.0),
                "I2": branch(1.0),
            }
            if index < 2:
                branches["I2"] = branch(1.0, passed=False)
            records.append({"anchor_id": str(index), "branches": branches})
        result = analyze_pilot(records)
        self.assertFalse(result["criteria"]["alignment_at_least_90pct"])


if __name__ == "__main__":
    unittest.main()
