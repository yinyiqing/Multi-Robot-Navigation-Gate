import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from actor_gradient_guard import actor_gradient_gate_decision


class ActorGradientGuardTests(unittest.TestCase):
    def test_rejects_too_few_dangerous_samples(self):
        passed, _ = actor_gradient_gate_decision(31, 0.5, 0.5, 32, 0.9, 0.9)
        self.assertFalse(passed)

    def test_rejects_uniform_positive_gradients(self):
        passed, angular_share = actor_gradient_gate_decision(
            100, 0.95, 0.99, 32, 0.9, 0.9
        )
        self.assertFalse(passed)
        self.assertAlmostEqual(angular_share, 0.99)

    def test_accepts_balanced_angular_and_nonuniform_linear_gradients(self):
        passed, angular_share = actor_gradient_gate_decision(
            100, 0.65, 0.52, 32, 0.9, 0.9
        )
        self.assertTrue(passed)
        self.assertAlmostEqual(angular_share, 0.52)


if __name__ == "__main__":
    unittest.main()
