import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from multi_agent_velodyne_env import MultiAgentGazeboEnv


class MultiAgentRewardTests(unittest.TestCase):
    @staticmethod
    def environment(forward_weight=0.5, stagnation_weight=0.03):
        environment = MultiAgentGazeboEnv.__new__(MultiAgentGazeboEnv)
        environment.forward_reward_weight = forward_weight
        environment.stagnation_penalty_weight = stagnation_weight
        return environment

    def test_default_reward_preserves_forward_and_stagnation_terms(self):
        environment = self.environment()
        moving = environment.get_reward(False, False, [0.4, 0.0], 2.0, 0.0)
        stopped = environment.get_reward(False, False, [0.0, 0.0], 2.0, 0.0)
        self.assertAlmostEqual(moving, 0.2)
        self.assertAlmostEqual(stopped, -0.03)

    def test_strong_interaction_reward_removes_speed_bias(self):
        environment = self.environment(forward_weight=0.0, stagnation_weight=0.0)
        moving = environment.get_reward(False, False, [0.4, 0.0], 2.0, 0.0)
        stopped = environment.get_reward(False, False, [0.0, 0.0], 2.0, 0.0)
        self.assertAlmostEqual(moving, 0.0)
        self.assertAlmostEqual(stopped, 0.0)

    def test_terminal_rewards_do_not_depend_on_shaping(self):
        environment = self.environment(forward_weight=0.0, stagnation_weight=0.0)
        self.assertEqual(
            environment.get_reward(True, False, [0.0, 0.0], 0.1, 0.0), 100.0
        )
        self.assertEqual(
            environment.get_reward(False, True, [0.0, 0.0], 0.1, 0.0), -100.0
        )

    def test_robot_proximity_penalty_is_zero_outside_safe_distance(self):
        environment = self.environment()
        environment.robot_safe_distance = 1.2
        environment.robot_proximity_penalty_weight = 5.0
        environment.robot_proximity_speed_penalty_weight = 10.0
        self.assertEqual(
            environment._compute_robot_proximity_penalty([1.0, 0.0], 1.2),
            0.0,
        )

    def test_robot_proximity_penalty_makes_fast_approach_more_expensive(self):
        environment = self.environment()
        environment.robot_safe_distance = 1.2
        environment.robot_proximity_penalty_weight = 5.0
        environment.robot_proximity_speed_penalty_weight = 10.0
        stopped = environment._compute_robot_proximity_penalty([0.0, 0.0], 0.8)
        moving = environment._compute_robot_proximity_penalty([1.0, 0.0], 0.8)
        self.assertAlmostEqual(stopped, 2.0)
        self.assertAlmostEqual(moving, 2.0 + 10.0 / 3.0)

    def test_robot_proximity_penalty_preserves_legacy_formula_by_default(self):
        environment = self.environment()
        environment.robot_safe_distance = 1.0
        environment.robot_proximity_penalty_weight = 5.0
        environment.robot_proximity_speed_penalty_weight = 0.0
        self.assertAlmostEqual(
            environment._compute_robot_proximity_penalty([1.0, 0.0], 0.8),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
