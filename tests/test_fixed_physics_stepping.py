import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from multi_agent_velodyne_env import MultiAgentGazeboEnv


class FixedPhysicsSteppingTests(unittest.TestCase):
    @staticmethod
    def environment():
        environment = MultiAgentGazeboEnv.__new__(MultiAgentGazeboEnv)
        environment.fixed_physics_step_size = 0.001
        environment.fixed_step_world_service = mock.Mock(
            return_value=SimpleNamespace(
                success=True,
                status_message="ok",
                sim_time=10.2,
            )
        )
        environment.sim_clock_time = 10.0
        environment._pause_fixed_physics = mock.Mock()
        environment._wait_for_stable_sim_time = mock.Mock(
            side_effect=[10.0, 10.2]
        )
        environment._get_gazebo_sim_time = mock.Mock(return_value=10.0)
        environment._wait_for_sim_time_at_least = mock.Mock(return_value=10.2)
        return environment

    @mock.patch("multi_agent_velodyne_env.subprocess.run")
    def test_service_request_does_not_spawn_gazebo_cli(self, run):
        environment = self.environment()
        actual = environment._request_fixed_physics_steps(200)
        self.assertAlmostEqual(actual, 10.2)
        environment.fixed_step_world_service.assert_called_once_with(200)
        run.assert_not_called()

    def test_service_rejection_is_not_silently_accepted(self):
        environment = self.environment()
        environment.fixed_step_world_service.return_value = SimpleNamespace(
            success=False,
            status_message="world is not paused",
            sim_time=10.0,
        )
        with self.assertRaisesRegex(RuntimeError, "world is not paused"):
            environment._request_fixed_physics_steps(200)

    def test_advance_uses_exact_service_step_and_finishes_paused(self):
        environment = self.environment()
        actual = environment._advance_fixed_physics(0.2)
        self.assertAlmostEqual(actual, 10.2)
        environment.fixed_step_world_service.assert_called_once_with(200)
        self.assertEqual(environment._pause_fixed_physics.call_count, 2)

    def test_zero_progress_is_rejected_after_bounded_retries(self):
        environment = self.environment()
        environment._wait_for_stable_sim_time = mock.Mock(return_value=10.0)
        environment._wait_for_sim_time_at_least = mock.Mock(
            side_effect=TimeoutError("no progress")
        )
        with self.assertRaisesRegex(TimeoutError, "did not complete 200"):
            environment._advance_fixed_physics(0.2)
        self.assertEqual(environment.fixed_step_world_service.call_count, 3)

    def test_overshoot_is_rejected(self):
        environment = self.environment()
        environment._wait_for_stable_sim_time = mock.Mock(
            side_effect=[10.0, 10.25]
        )
        with self.assertRaisesRegex(RuntimeError, "overshot target"):
            environment._advance_fixed_physics(0.2)
        self.assertEqual(environment._pause_fixed_physics.call_count, 2)

    @mock.patch("multi_agent_velodyne_env.subprocess.run")
    def test_cli_compatibility_path_checks_return_code(self, run):
        environment = self.environment()
        environment.fixed_step_world_service = None
        run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="transport unavailable"
        )
        with self.assertRaisesRegex(RuntimeError, "transport unavailable"):
            environment._request_fixed_physics_steps(200)

    @mock.patch("multi_agent_velodyne_env.StepWorld", None)
    def test_required_service_cannot_silently_fall_back_to_cli(self):
        environment = self.environment()
        environment.fixed_agent_interfaces_ready = False
        environment.require_fixed_step_service = True
        with self.assertRaisesRegex(RuntimeError, "bindings are unavailable"):
            environment._wait_for_fixed_agent_interfaces()


if __name__ == "__main__":
    unittest.main()
