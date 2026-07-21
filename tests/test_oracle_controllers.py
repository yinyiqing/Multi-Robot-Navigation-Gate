import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from oracle_controllers import ConflictPairYieldOracle


class ConstantPolicy:
    def get_action(self, state):
        return np.array([0.5, -0.25], dtype=np.float32)


class FakeEnv:
    def __init__(self):
        self.current_curriculum_case = {
            "metrics": {"conflict_edges": [{"agents": ["r1", "r2"]}]}
        }
        self.robot_positions = {
            "r1": np.array([0.0, 0.0]),
            "r2": np.array([2.0, 0.0]),
        }


class ConflictPairYieldOracleTests(unittest.TestCase):
    def setUp(self):
        self.env = FakeEnv()
        self.oracle = ConflictPairYieldOracle(ConstantPolicy(), max_yield_steps=2)

    def test_higher_named_agent_yields_inside_stop_distance(self):
        self.env.robot_positions["r2"] = np.array([1.1, 0.0])
        passing_action, passing_yield = self.oracle.choose_action(
            self.env, "r1", np.zeros(2), {"r1", "r2"}
        )
        yielding_action, yielding = self.oracle.choose_action(
            self.env, "r2", np.zeros(2), {"r1", "r2"}
        )
        np.testing.assert_allclose(passing_action, [0.5, -0.25])
        self.assertFalse(passing_yield)
        np.testing.assert_allclose(yielding_action, [-1.0, 0.0])
        self.assertTrue(yielding)

    def test_yielder_releases_after_pair_separates(self):
        self.env.robot_positions["r2"] = np.array([1.1, 0.0])
        self.oracle.choose_action(self.env, "r2", np.zeros(2), {"r1", "r2"})
        self.env.robot_positions["r2"] = np.array([1.5, 0.0])
        action, yielding = self.oracle.choose_action(
            self.env, "r2", np.zeros(2), {"r1", "r2"}
        )
        np.testing.assert_allclose(action, [0.5, -0.25])
        self.assertFalse(yielding)

    def test_yielder_releases_when_passer_is_done(self):
        self.env.robot_positions["r2"] = np.array([1.1, 0.0])
        self.oracle.choose_action(self.env, "r2", np.zeros(2), {"r1", "r2"})
        action, yielding = self.oracle.choose_action(
            self.env, "r2", np.zeros(2), {"r2"}
        )
        np.testing.assert_allclose(action, [0.5, -0.25])
        self.assertFalse(yielding)

    def test_invalid_thresholds_are_rejected(self):
        with self.assertRaises(ValueError):
            ConflictPairYieldOracle(ConstantPolicy(), stop_distance=1.2, release_distance=1.0)


if __name__ == "__main__":
    unittest.main()
