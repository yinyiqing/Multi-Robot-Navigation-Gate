import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from actor_counterfactual import (
    LABEL_AMBIGUOUS,
    LABEL_GENERALIST,
    LABEL_STRONG,
    choose_actor_label,
    counterfactual_repeatability,
)


def outcome(**overrides):
    values = {
        "ego_collision": False,
        "collision_count": 0,
        "ego_target": False,
        "minimum_ego_clearance": 1.0,
        "ego_progress": 0.10,
    }
    values.update(overrides)
    return values


class CounterfactualLabelTests(unittest.TestCase):
    def test_collision_has_priority_over_progress(self):
        label, reason = choose_actor_label(
            outcome(ego_collision=True, collision_count=1, ego_progress=0.4),
            outcome(ego_progress=0.0),
        )
        self.assertEqual((label, reason), (LABEL_STRONG, "ego_collision"))

    def test_clearance_gain_with_large_progress_regression_is_ambiguous(self):
        label, reason = choose_actor_label(
            outcome(minimum_ego_clearance=0.8, ego_progress=0.20),
            outcome(minimum_ego_clearance=1.0, ego_progress=0.05),
        )
        self.assertEqual((label, reason), (LABEL_AMBIGUOUS, "ambiguous"))

    def test_clearance_gain_selects_strong_when_progress_is_preserved(self):
        label, reason = choose_actor_label(
            outcome(minimum_ego_clearance=0.8, ego_progress=0.15),
            outcome(minimum_ego_clearance=0.95, ego_progress=0.12),
        )
        self.assertEqual((label, reason), (LABEL_STRONG, "clearance"))

    def test_small_differences_remain_ambiguous(self):
        label, reason = choose_actor_label(
            outcome(minimum_ego_clearance=0.90, ego_progress=0.10),
            outcome(minimum_ego_clearance=0.95, ego_progress=0.12),
        )
        self.assertEqual((label, reason), (LABEL_AMBIGUOUS, "ambiguous"))

    def test_repeatability_rejects_discrete_or_large_metric_changes(self):
        stable = counterfactual_repeatability(
            outcome(minimum_ego_clearance=0.90, ego_progress=0.10),
            outcome(minimum_ego_clearance=0.92, ego_progress=0.12),
        )
        self.assertTrue(stable["repeatable"])
        unstable = counterfactual_repeatability(
            outcome(), outcome(ego_collision=True, collision_count=1)
        )
        self.assertFalse(unstable["repeatable"])

    def test_repeatability_ignores_terminal_progress_after_both_reach(self):
        result = counterfactual_repeatability(
            outcome(ego_target=True, ego_progress=0.70),
            outcome(ego_target=True, ego_progress=0.62),
        )
        self.assertTrue(result["repeatable"])
        self.assertAlmostEqual(result["progress_delta"], 0.08)


if __name__ == "__main__":
    unittest.main()
