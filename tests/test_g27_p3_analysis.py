import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_g27_p3_validation import aggregate, paired_effect


def rows(full_success, collisions, timeout, steps, share):
    result = np.zeros((2, 17), dtype=object)
    result[:, 8] = full_success
    result[:, 7] = collisions
    result[:, 11] = timeout
    result[:, 3] = steps
    result[:, 14] = share
    result[:, 15] = [1, 2]
    return result


class G27P3AnalysisTest(unittest.TestCase):
    def test_robot_collision_uses_five_robot_denominator(self):
        value = rows([1, 0], [1, 2], [0, 1], [10, 20], [0.4, 0.6])
        self.assertAlmostEqual(aggregate(value)["robot_collision"], 0.3)

    def test_paired_success_steps_only_uses_joint_success(self):
        baseline = rows([1, 0], [0, 0], [0, 0], [20, 100], [0.7, 0.7])
        candidate = rows([1, 1], [0, 0], [0, 0], [14, 30], [0.5, 0.5])
        effect = paired_effect(candidate, baseline)
        self.assertEqual(effect["paired_success_steps"]["pairs"], 1)
        self.assertEqual(effect["paired_success_steps"]["mean_difference"], -6.0)
        self.assertAlmostEqual(
            effect["interaction_selection_share"]["mean_difference"], -0.2
        )


if __name__ == "__main__":
    unittest.main()
