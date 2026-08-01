import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from actor_counterfactual import (
    LABEL_AMBIGUOUS,
    LABEL_GENERALIST,
    LABEL_STRONG,
    bootstrap_mean_difference_interval,
    choose_actor_label,
    choose_actor_distribution_label,
    counterfactual_repeatability,
    distribution_label_repeatability,
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

    def test_distribution_label_selects_lower_collision_actor(self):
        generalist = [outcome(ego_collision=True, collision_count=1) for _ in range(5)]
        strong = [outcome() for _ in range(5)]
        label, reason, diagnostics = choose_actor_distribution_label(
            generalist, strong, resamples=1000, seed=7
        )
        self.assertEqual((label, reason), (LABEL_STRONG, "ego_collision_rate"))
        self.assertLess(diagnostics["ego_collision"]["interval"][1], 0.0)

    def test_distribution_label_keeps_overlapping_outcomes_ambiguous(self):
        generalist = [
            outcome(minimum_ego_clearance=value, ego_progress=0.10)
            for value in (0.80, 0.95, 0.85, 0.90, 1.00)
        ]
        strong = [
            outcome(minimum_ego_clearance=value, ego_progress=0.11)
            for value in (0.85, 1.00, 0.90, 0.95, 1.05)
        ]
        label, reason, _ = choose_actor_distribution_label(
            generalist, strong, resamples=1000, seed=9
        )
        self.assertEqual((label, reason), (LABEL_AMBIGUOUS, "ambiguous_distribution"))

    def test_distribution_label_requires_multiple_rollouts(self):
        with self.assertRaisesRegex(ValueError, "at least two rollouts"):
            choose_actor_distribution_label([outcome()], [outcome()])

    def test_bootstrap_interval_is_deterministic(self):
        first = bootstrap_mean_difference_interval(
            [0.0, 0.1, 0.2], [0.4, 0.5, 0.6], resamples=1000, seed=3
        )
        second = bootstrap_mean_difference_interval(
            [0.0, 0.1, 0.2], [0.4, 0.5, 0.6], resamples=1000, seed=3
        )
        self.assertEqual(first, second)
        self.assertGreater(first["interval"][0], 0.0)

    def test_distribution_batch_labels_must_agree(self):
        stable = distribution_label_repeatability([LABEL_STRONG, LABEL_STRONG])
        disagreement = distribution_label_repeatability(
            [LABEL_STRONG, LABEL_GENERALIST]
        )
        ambiguous = distribution_label_repeatability(
            [LABEL_AMBIGUOUS, LABEL_AMBIGUOUS]
        )
        self.assertTrue(stable["repeatable"])
        self.assertEqual(stable["label"], LABEL_STRONG)
        self.assertFalse(disagreement["repeatable"])
        self.assertFalse(ambiguous["repeatable"])


if __name__ == "__main__":
    unittest.main()
