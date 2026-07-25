import math
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from multi_agent_velodyne_env import MultiAgentGazeboEnv


class NeighborContextTests(unittest.TestCase):
    @staticmethod
    def environment(mode, positions, yaws):
        environment = MultiAgentGazeboEnv.__new__(MultiAgentGazeboEnv)
        environment.agent_names = ["r1", "r2"]
        environment.active_neighbors_only = False
        environment.neighbor_context_mode = mode
        environment.robot_positions = {
            name: np.asarray(position, dtype=np.float32)
            for name, position in positions.items()
        }
        environment._get_robot_yaw = lambda name: yaws[name]
        environment._compute_visible_neighbors = (
            lambda name, active_names=None: ["r2"] if name == "r1" else ["r1"]
        )
        return environment

    def test_ego_motion_context_uses_local_position_and_relative_velocity(self):
        environment = self.environment(
            "ego_motion",
            {"r1": [1.0, 2.0], "r2": [2.0, 2.0]},
            {"r1": math.pi / 2.0, "r2": 0.0},
        )
        contexts = environment.build_neighbor_context(
            [[1.0, 0.0], [0.5, 0.0]], max_neighbors=1
        )
        np.testing.assert_allclose(
            contexts[0],
            [0.0, -1.0, 1.0, -math.pi / 2.0, -1.0, -0.5, 1.0],
            atol=1e-6,
        )

    def test_ego_motion_context_is_invariant_to_global_rotation(self):
        actions = [[0.8, 0.1], [0.4, -0.2]]
        base = self.environment(
            "ego_motion",
            {"r1": [0.0, 0.0], "r2": [2.0, 1.0]},
            {"r1": 0.3, "r2": -0.4},
        ).build_neighbor_context(actions, max_neighbors=1)

        angle = 1.1
        rotation = np.array(
            [[math.cos(angle), -math.sin(angle)],
             [math.sin(angle), math.cos(angle)]],
            dtype=np.float32,
        )
        rotated = self.environment(
            "ego_motion",
            {
                "r1": rotation @ np.array([0.0, 0.0]),
                "r2": rotation @ np.array([2.0, 1.0]),
            },
            {"r1": 0.3 + angle, "r2": -0.4 + angle},
        ).build_neighbor_context(actions, max_neighbors=1)
        np.testing.assert_allclose(base, rotated, atol=1e-6)

    def test_legacy_context_preserves_world_frame_position(self):
        environment = self.environment(
            "legacy",
            {"r1": [1.0, 2.0], "r2": [2.0, 2.0]},
            {"r1": math.pi / 2.0, "r2": 0.0},
        )
        contexts = environment.build_neighbor_context(
            [[0.0, 0.0], [0.0, 0.0]], max_neighbors=1, include_actions=False
        )
        np.testing.assert_allclose(
            contexts[0], [1.0, 0.0, 1.0, -math.pi / 2.0, 1.0], atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
