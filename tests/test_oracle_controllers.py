import sys
import types
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from oracle_controllers import ConflictPairYieldOracle, RightHandPassOracle


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
        self.robot_yaws = {"r1": 0.0, "r2": np.pi}
        self.last_odom = {
            "r1": self._make_odom(0.5),
            "r2": self._make_odom(0.5),
        }

    @staticmethod
    def _make_odom(speed):
        linear = types.SimpleNamespace(x=float(speed))
        twist = types.SimpleNamespace(linear=linear)
        return types.SimpleNamespace(twist=types.SimpleNamespace(twist=twist))

    def _get_robot_yaw(self, name):
        return self.robot_yaws[name]


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


class RightHandPassOracleTests(unittest.TestCase):
    def setUp(self):
        self.env = FakeEnv()
        self.oracle = RightHandPassOracle(
            ConstantPolicy(),
            activation_distance=1.5,
            release_distance=1.8,
            turn_action=-0.6,
            linear_speed_cap=0.45,
            max_override_steps=2,
        )
        self.oracle.reset(["r1", "r2"])

    def test_both_head_on_agents_turn_to_their_own_right(self):
        self.env.robot_positions["r2"] = np.array([1.2, 0.0])
        first_action, first_active = self.oracle.choose_action(
            self.env, "r1", np.zeros(2), {"r1", "r2"}
        )
        second_action, second_active = self.oracle.choose_action(
            self.env, "r2", np.zeros(2), {"r1", "r2"}
        )
        np.testing.assert_allclose(first_action, [-0.1, -0.6])
        np.testing.assert_allclose(second_action, [-0.1, -0.6])
        self.assertTrue(first_active)
        self.assertTrue(second_active)

    def test_same_direction_agents_keep_base_action(self):
        self.env.robot_positions["r2"] = np.array([1.2, 0.0])
        self.env.robot_yaws["r2"] = 0.0
        action, active = self.oracle.choose_action(
            self.env, "r1", np.zeros(2), {"r1", "r2"}
        )
        np.testing.assert_allclose(action, [0.5, -0.25])
        self.assertFalse(active)

    def test_stationary_head_on_agents_keep_base_action(self):
        self.env.robot_positions["r2"] = np.array([1.2, 0.0])
        self.env.last_odom["r1"] = self.env._make_odom(0.0)
        self.env.last_odom["r2"] = self.env._make_odom(0.0)
        action, active = self.oracle.choose_action(
            self.env, "r1", np.zeros(2), {"r1", "r2"}
        )
        np.testing.assert_allclose(action, [0.5, -0.25])
        self.assertFalse(active)

    def test_max_override_does_not_immediately_reactivate(self):
        self.env.robot_positions["r2"] = np.array([1.2, 0.0])
        for _ in range(2):
            _, active = self.oracle.choose_action(
                self.env, "r1", np.zeros(2), {"r1", "r2"}
            )
            self.assertTrue(active)
        action, active = self.oracle.choose_action(
            self.env, "r1", np.zeros(2), {"r1", "r2"}
        )
        np.testing.assert_allclose(action, [0.5, -0.25])
        self.assertFalse(active)
        _, active = self.oracle.choose_action(
            self.env, "r1", np.zeros(2), {"r1", "r2"}
        )
        self.assertFalse(active)

    def test_rule_releases_after_agents_separate(self):
        self.env.robot_positions["r2"] = np.array([1.2, 0.0])
        self.oracle.choose_action(self.env, "r1", np.zeros(2), {"r1", "r2"})
        self.env.robot_positions["r2"] = np.array([2.0, 0.0])
        action, active = self.oracle.choose_action(
            self.env, "r1", np.zeros(2), {"r1", "r2"}
        )
        np.testing.assert_allclose(action, [0.5, -0.25])
        self.assertFalse(active)

    def test_invalid_right_hand_thresholds_are_rejected(self):
        with self.assertRaises(ValueError):
            RightHandPassOracle(
                ConstantPolicy(), activation_distance=1.5, release_distance=1.0
            )


if __name__ == "__main__":
    unittest.main()
