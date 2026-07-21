import math
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from temporal_interaction import TemporalInteractionEncoder, build_front_lidar_gaps


class TemporalInteractionEncoderTests(unittest.TestCase):
    def setUp(self):
        self.gaps = build_front_lidar_gaps(20)
        self.encoder = TemporalInteractionEncoder(self.gaps)
        self.index = 10
        self.angle = float(np.mean(self.gaps[self.index]))

    def scan_with_point(self, index, distance):
        scan = np.full(20, 10.0, dtype=np.float32)
        scan[index] = distance
        return scan

    def project_static_point(self, distance, previous_pose, current_pose):
        previous_x, previous_y, previous_yaw = previous_pose
        current_x, current_y, current_yaw = current_pose
        local = np.array(
            [distance * math.cos(self.angle), distance * math.sin(self.angle)]
        )
        rotation = np.array(
            [
                [math.cos(previous_yaw), -math.sin(previous_yaw)],
                [math.sin(previous_yaw), math.cos(previous_yaw)],
            ]
        )
        world = np.array([previous_x, previous_y]) + rotation @ local
        delta = world - np.array([current_x, current_y])
        inverse = np.array(
            [
                [math.cos(current_yaw), math.sin(current_yaw)],
                [-math.sin(current_yaw), math.cos(current_yaw)],
            ]
        )
        current = inverse @ delta
        angle = math.atan2(current[1], current[0])
        matches = np.flatnonzero(
            (self.gaps[:, 0] <= angle) & (angle < self.gaps[:, 1])
        )
        return int(matches[0]), float(np.linalg.norm(current))

    def test_first_frame_is_neutral(self):
        result = self.encoder.update(
            self.scan_with_point(self.index, 2.0), [0.0, 0.0, 0.0], 0.0
        )
        np.testing.assert_allclose(result["features"], 0.0)
        self.assertFalse(np.any(result["valid"]))

    def test_static_point_is_stationary_after_translation_compensation(self):
        previous_pose = [0.0, 0.0, 0.0]
        current_pose = [0.1, 0.0, 0.0]
        self.encoder.update(
            self.scan_with_point(self.index, 2.0), previous_pose, 0.0
        )
        index, distance = self.project_static_point(2.0, previous_pose, current_pose)
        result = self.encoder.update(
            self.scan_with_point(index, distance), current_pose, 0.2
        )
        self.assertAlmostEqual(float(result["closing_speed"][index]), 0.0, places=5)
        self.assertAlmostEqual(float(result["urgency"][index]), 0.0, places=5)

    def test_moving_point_produces_closing_speed_and_ttc(self):
        previous_pose = [0.0, 0.0, 0.0]
        current_pose = [0.1, 0.0, 0.0]
        self.encoder.update(
            self.scan_with_point(self.index, 2.0), previous_pose, 0.0
        )
        index, static_distance = self.project_static_point(
            2.0, previous_pose, current_pose
        )
        current_distance = static_distance - 0.1
        result = self.encoder.update(
            self.scan_with_point(index, current_distance), current_pose, 0.2
        )
        self.assertAlmostEqual(float(result["closing_speed"][index]), 0.5, places=4)
        expected_ttc = (current_distance - 0.35) / 0.5
        self.assertAlmostEqual(float(result["ttc"][index]), expected_ttc, places=4)
        self.assertGreater(float(result["urgency"][index]), 0.0)

    def test_rotation_is_compensated(self):
        previous_pose = [0.0, 0.0, 0.0]
        current_pose = [0.0, 0.0, 0.08]
        self.encoder.update(
            self.scan_with_point(self.index, 2.0), previous_pose, 0.0
        )
        index, distance = self.project_static_point(2.0, previous_pose, current_pose)
        result = self.encoder.update(
            self.scan_with_point(index, distance), current_pose, 0.2
        )
        self.assertAlmostEqual(float(result["closing_speed"][index]), 0.0, places=5)

    def test_timestamp_must_increase(self):
        scan = self.scan_with_point(self.index, 2.0)
        self.encoder.update(scan, [0.0, 0.0, 0.0], 1.0)
        with self.assertRaisesRegex(ValueError, "increase strictly"):
            self.encoder.update(scan, [0.0, 0.0, 0.0], 1.0)


if __name__ == "__main__":
    unittest.main()
