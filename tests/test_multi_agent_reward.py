import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


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

    def test_robot_threat_suppresses_forward_bonus_but_keeps_progress(self):
        environment = self.environment(forward_weight=0.5, stagnation_weight=0.0)
        reward = environment.get_reward(
            False,
            False,
            [0.8, 0.0],
            2.0,
            0.02,
            suppress_forward_reward=True,
        )
        self.assertAlmostEqual(reward, 0.4)

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

    @staticmethod
    def clearance_environment():
        environment = MultiAgentGazeboEnv.__new__(MultiAgentGazeboEnv)
        environment.robot_safe_distance = 1.2
        environment.robot_clearance_reward_weight = 20.0
        environment.robot_clearance_reward_max_gain = 0.1
        return environment

    def test_robot_clearance_rewards_safe_separation_with_goal_progress(self):
        environment = self.clearance_environment()
        reward = environment._compute_robot_clearance_reward(
            False, False, 0.8, 0.86
        )
        self.assertAlmostEqual(reward, 1.2)

    def test_robot_clearance_rewards_separation_while_yielding(self):
        environment = self.clearance_environment()
        reward = environment._compute_robot_clearance_reward(
            False, False, 0.8, 0.9
        )
        self.assertAlmostEqual(reward, 2.0)

    def test_robot_clearance_penalizes_worsening_separation(self):
        environment = self.clearance_environment()
        reward = environment._compute_robot_clearance_reward(
            False, False, 0.9, 0.84
        )
        self.assertAlmostEqual(reward, -1.2)

    def test_robot_clearance_is_neutral_when_distance_is_unchanged(self):
        environment = self.clearance_environment()
        reward = environment._compute_robot_clearance_reward(
            False, False, 0.8, 0.8
        )
        self.assertEqual(reward, 0.0)

    def test_robot_clearance_reward_is_capped_per_step(self):
        environment = self.clearance_environment()
        reward = environment._compute_robot_clearance_reward(
            False, False, 0.7, 1.1
        )
        self.assertAlmostEqual(reward, 2.0)

    def test_robot_clearance_negative_reward_is_capped_per_step(self):
        environment = self.clearance_environment()
        reward = environment._compute_robot_clearance_reward(
            False, False, 1.1, 0.7
        )
        self.assertAlmostEqual(reward, -2.0)

    def test_robot_threat_suppresses_stagnation_penalties(self):
        environment = self.environment()
        environment.anti_stagnation_reward = True
        environment.anti_stagnation_penalty = 0.1
        environment.anti_stagnation_linear_threshold = 0.05
        environment.anti_stagnation_progress_threshold = 0.005
        environment.anti_stagnation_min_laser = 0.35
        environment.robot_safe_distance = 1.2

        base_reward = environment.get_reward(
            False,
            False,
            [0.0, 0.0],
            0.8,
            0.0,
            suppress_stagnation=True,
        )
        extra_penalty = environment._compute_anti_stagnation_penalty(
            False, False, [0.0, 0.0], 0.8, 0.0, 0.9
        )
        self.assertAlmostEqual(base_reward, -0.1)
        self.assertEqual(extra_penalty, 0.0)

    def test_stagnation_penalties_resume_without_robot_threat(self):
        environment = self.environment()
        environment.anti_stagnation_reward = True
        environment.anti_stagnation_penalty = 0.1
        environment.anti_stagnation_linear_threshold = 0.05
        environment.anti_stagnation_progress_threshold = 0.005
        environment.anti_stagnation_min_laser = 0.35
        environment.robot_safe_distance = 1.2

        base_reward = environment.get_reward(
            False, False, [0.0, 0.0], 0.8, 0.0
        )
        extra_penalty = environment._compute_anti_stagnation_penalty(
            False, False, [0.0, 0.0], 0.8, 0.0, 1.3
        )
        self.assertAlmostEqual(base_reward, -0.13)
        self.assertAlmostEqual(extra_penalty, 0.1)

    @staticmethod
    def recovery_environment():
        environment = MultiAgentGazeboEnv.__new__(MultiAgentGazeboEnv)
        environment.safe_recovery_reward = True
        environment.safe_recovery_penalty = 0.2
        environment.safe_recovery_linear_threshold = 0.25
        environment.safe_recovery_progress_threshold = 0.003
        environment.safe_recovery_min_laser = 0.6
        environment.safe_recovery_robot_distance = 1.2
        environment.safe_recovery_progress_bonus_weight = 0.8
        environment.safe_recovery_idle_penalty_weight = 1.0
        return environment

    def test_safe_recovery_penalizes_low_motion_when_clear(self):
        environment = self.recovery_environment()
        penalty = environment._compute_safe_recovery_penalty(
            False, False, [0.1, 0.0], 0.8, 0.001, 1.5
        )
        self.assertAlmostEqual(penalty, 0.35)

    def test_safe_recovery_does_not_penalize_robot_threat_waiting(self):
        environment = self.recovery_environment()
        penalty = environment._compute_safe_recovery_penalty(
            False, False, [0.1, 0.0], 0.8, 0.001, 0.9
        )
        self.assertEqual(penalty, 0.0)

    def test_safe_recovery_does_not_penalize_forward_progress(self):
        environment = self.recovery_environment()
        penalty = environment._compute_safe_recovery_penalty(
            False, False, [0.4, 0.0], 0.8, 0.001, 1.5
        )
        self.assertEqual(penalty, 0.0)

    def test_safe_recovery_rewards_clear_forward_progress(self):
        environment = self.recovery_environment()
        reward = environment._compute_safe_recovery_reward(
            False, False, [0.4, 0.0], 0.8, 0.009, 1.5
        )
        self.assertAlmostEqual(reward, 0.8)

    def test_safety_distance_uses_only_critic_visible_active_neighbors(self):
        environment = MultiAgentGazeboEnv.__new__(MultiAgentGazeboEnv)
        environment.robot_positions = {
            "r1": np.array([0.0, 0.0]),
            "r2": np.array([0.8, 0.0]),
            "r3": np.array([0.4, 0.0]),
        }
        environment._compute_visible_neighbors = lambda name, active_names: [
            other for other in ("r2",) if other in active_names
        ]
        self.assertAlmostEqual(
            environment._nearest_visible_robot_distance("r1", {"r1", "r2"}),
            0.8,
        )
        self.assertEqual(
            environment._nearest_visible_robot_distance("r1", {"r1"}),
            float("inf"),
        )

    @staticmethod
    def yield_environment():
        environment = MultiAgentGazeboEnv.__new__(MultiAgentGazeboEnv)
        environment.agent_names = ["r1", "r2"]
        environment.robot_positions = {
            "r1": np.array([0.0, 0.0]),
            "r2": np.array([0.8, 0.0]),
        }
        environment.goal_positions = {
            "r1": np.array([3.0, 0.0]),
            "r2": np.array([1.0, 0.0]),
        }
        environment.reward_neighbor_radius = 10.0
        environment.reward_neighbor_fov = np.pi
        environment._get_robot_yaw = lambda name: 0.0
        environment.robot_safe_distance = 1.2
        environment.yield_priority_reward = True
        environment.yield_priority_distance = 1.0
        environment.yield_priority_goal_margin = 0.25
        environment.yield_priority_stop_linear = 0.25
        environment.yield_priority_penalty_weight = 4.0
        environment.yield_priority_bonus_weight = 1.0
        environment.yield_priority_clearance_bonus_weight = 2.0
        environment.yield_priority_restart_bonus_weight = 1.5
        environment.yield_priority_stale_wait_penalty_weight = 0.5
        environment.yield_priority_max_wait_steps = 8
        environment.emergency_stop_distance = 0.0
        environment.emergency_stop_penalty_weight = 8.0
        environment.emergency_stop_bonus_weight = 1.0
        environment.robot_clearance_reward_max_gain = 0.1
        environment.yield_wait_steps = {"r1": 0, "r2": 0}
        environment.yield_nearest_distances = {"r1": None, "r2": None}
        return environment

    def test_yield_priority_penalizes_farther_robot_for_pushing_forward(self):
        environment = self.yield_environment()
        reward = environment._compute_yield_priority_reward(
            "r1", [1.0, 0.0], 3.0, {"r1", "r2"}
        )
        self.assertAlmostEqual(reward, -0.8)

    def test_yield_priority_rewards_farther_robot_for_waiting(self):
        environment = self.yield_environment()
        reward = environment._compute_yield_priority_reward(
            "r1", [0.0, 0.0], 3.0, {"r1", "r2"}
        )
        self.assertAlmostEqual(reward, 0.2)

    def test_yield_priority_does_not_penalize_nearer_robot(self):
        environment = self.yield_environment()
        reward = environment._compute_yield_priority_reward(
            "r2", [1.0, 0.0], 0.2, {"r1", "r2"}
        )
        self.assertEqual(reward, 0.0)

    def test_emergency_stop_penalizes_all_fast_close_robots(self):
        environment = self.yield_environment()
        environment.emergency_stop_distance = 0.9
        reward = environment._compute_yield_priority_reward(
            "r2", [1.0, 0.0], 0.2, {"r1", "r2"}
        )
        self.assertAlmostEqual(reward, -8.0 / 9.0)

    def test_yield_priority_rewards_waiting_that_increases_clearance(self):
        environment = self.yield_environment()
        environment.yield_nearest_distances["r1"] = 0.7
        reward = environment._compute_yield_priority_reward(
            "r1", [0.0, 0.0], 3.0, {"r1", "r2"}
        )
        # Base waiting reward is 0.2; extra clearance gain is capped at 0.1
        # and weighted by 2.0.
        self.assertAlmostEqual(reward, 0.4)

    def test_yield_priority_rewards_restart_after_safe_wait(self):
        environment = self.yield_environment()
        environment.robot_positions["r2"] = np.array([1.4, 0.0])
        environment.yield_wait_steps["r1"] = 4
        reward = environment._compute_yield_priority_reward(
            "r1", [0.25, 0.0], 3.0, {"r1", "r2"}
        )
        self.assertAlmostEqual(reward, 0.75)
        self.assertEqual(environment.yield_wait_steps["r1"], 0)

    def test_yield_priority_penalizes_stale_waiting_when_safe(self):
        environment = self.yield_environment()
        environment.robot_positions["r2"] = np.array([1.4, 0.0])
        environment.yield_wait_steps["r1"] = 4
        reward = environment._compute_yield_priority_reward(
            "r1", [0.0, 0.0], 3.0, {"r1", "r2"}
        )
        self.assertAlmostEqual(reward, -0.5)
        self.assertEqual(environment.yield_wait_steps["r1"], 0)


class ManifestPairedCycleTests(unittest.TestCase):
    @staticmethod
    def environment():
        environment = MultiAgentGazeboEnv.__new__(MultiAgentGazeboEnv)
        environment.scenario_mode = "manifest"
        environment.curriculum_cases = [
            {"scenario_id": "A"},
            {"scenario_id": "B"},
            {"scenario_id": "C"},
        ]
        environment.curriculum_case_index = 0
        environment.manifest_band_cases = {}
        environment.manifest_band_indices = {}
        environment.manifest_band_schedule_index = 0
        return environment

    @mock.patch.dict("os.environ", {"DRL_MULTI_MANIFEST_SAMPLING": "paired_cycle"})
    def test_each_manifest_case_is_selected_twice_in_order(self):
        environment = self.environment()
        selected = [
            environment._select_curriculum_case()["scenario_id"]
            for _ in range(6)
        ]
        self.assertEqual(selected, ["A", "A", "B", "B", "C", "C"])

    @mock.patch.dict("os.environ", {"DRL_MULTI_MANIFEST_SAMPLING": "paired_cycle"})
    def test_sampling_state_restores_the_second_half_of_a_pair(self):
        environment = self.environment()
        for _ in range(3):
            environment._select_curriculum_case()
        state = environment.manifest_sampling_state()

        restored = self.environment()
        restored.restore_manifest_sampling_state(state)
        self.assertEqual(
            restored._select_curriculum_case()["scenario_id"],
            "B",
        )

if __name__ == "__main__":
    unittest.main()
