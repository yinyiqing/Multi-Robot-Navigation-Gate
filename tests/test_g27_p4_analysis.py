import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/analyze_g27_p4_test.py"
SPEC = importlib.util.spec_from_file_location("analyze_g27_p4_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def comparisons(
    success_ci=(0.01, 0.10),
    success_p=0.01,
    collision_ci=(-0.10, -0.01),
    replacement_success=0.0,
    replacement_collision=0.02,
    replacement_timeout=0.01,
):
    return {
        "b3_minus_5a": {
            "full_success": {
                "scene_cluster_bca_95_ci": list(success_ci),
                "sign_flip_two_sided_p": success_p,
            },
            "robot_collision": {"scene_cluster_bca_95_ci": list(collision_ci)},
        },
        "b3_minus_b2": {
            "full_success": {"mean_difference": replacement_success},
            "robot_collision": {"mean_difference": replacement_collision},
            "episode_timeout": {"mean_difference": replacement_timeout},
            "interaction_selection_share": {"mean_difference": 1.0},
            "paired_success_steps": {"mean_difference": 100.0},
        },
    }


class ConfirmationCriteriaTest(unittest.TestCase):
    def test_boundary_values_pass_and_efficiency_is_not_a_criterion(self):
        criteria = MODULE.evaluate_confirmation_criteria(comparisons())
        self.assertTrue(all(criteria.values()))
        self.assertEqual(len(criteria), 5)
        self.assertNotIn("b3_efficiency_improves_vs_b2", criteria)

    def test_each_success_or_safety_failure_blocks_confirmation(self):
        cases = (
            {"success_ci": (0.0, 0.10)},
            {"success_p": 0.05},
            {"collision_ci": (-0.10, 0.0)},
            {"replacement_success": -1e-6},
            {"replacement_collision": 0.020001},
            {"replacement_timeout": 0.010001},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                criteria = MODULE.evaluate_confirmation_criteria(
                    comparisons(**overrides)
                )
                self.assertFalse(all(criteria.values()))


if __name__ == "__main__":
    unittest.main()
