import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from lidar_cluster_tracking import LidarClusterTracker, cluster_points


def compact_cluster(center):
    center = np.asarray(center, dtype=np.float64)
    offsets = np.array(
        [[-0.08, -0.04], [-0.08, 0.04], [0.0, -0.06], [0.0, 0.06], [0.08, 0.0]]
    )
    return center + offsets


class LidarClusterTrackingTests(unittest.TestCase):
    def test_compact_cluster_is_kept_and_wall_is_rejected(self):
        robot = compact_cluster([1.2, 0.1])
        wall = np.column_stack([np.full(30, 2.0), np.linspace(-1.0, 1.0, 30)])
        clusters = cluster_points(np.vstack([robot, wall]))
        self.assertEqual(len(clusters), 1)
        np.testing.assert_allclose(clusters[0]["centroid"], [1.2, 0.1], atol=0.03)

    def test_xyz_points_remain_compatible_with_xy_tracking(self):
        xy = compact_cluster([1.2, 0.1])
        xyz = np.column_stack([xy, np.linspace(-0.1, 0.3, len(xy))])
        clusters = cluster_points(xyz)
        self.assertEqual(len(clusters), 1)
        np.testing.assert_allclose(clusters[0]["centroid"], [1.2, 0.1], atol=0.03)

    def test_static_cluster_is_stationary_after_ego_motion_compensation(self):
        tracker = LidarClusterTracker()
        tracker.update(compact_cluster([2.0, 0.0]), [0.0, 0.0, 0.0], 0.0)
        tracks = tracker.update(
            compact_cluster([1.9, 0.0]), [0.1, 0.0, 0.0], 0.2
        )
        self.assertEqual(len(tracks), 1)
        np.testing.assert_allclose(tracks[0]["world_velocity"], 0.0, atol=1e-6)
        self.assertFalse(tracks[0]["urgent"])

    def test_crossing_cluster_produces_cpa_and_ttc(self):
        tracker = LidarClusterTracker(collision_distance=0.75)
        tracker.update(compact_cluster([1.2, 0.3]), [0.0, 0.0, 0.0], 0.0)
        tracks = tracker.update(
            compact_cluster([1.0, 0.2]), [0.0, 0.0, 0.0], 0.2
        )
        self.assertEqual(len(tracks), 1)
        track = tracks[0]
        np.testing.assert_allclose(track["velocity"], [-1.0, -0.5], atol=1e-6)
        self.assertLess(track["closest_approach_distance"], 0.3)
        self.assertLess(track["ttc"], 1.0)
        self.assertTrue(track["urgent"])

    def test_static_cluster_is_stationary_after_ego_rotation(self):
        tracker = LidarClusterTracker()
        tracker.update(compact_cluster([2.0, 0.0]), [0.0, 0.0, 0.0], 0.0)
        yaw = 0.1
        rotated_center = [2.0 * np.cos(yaw), -2.0 * np.sin(yaw)]
        tracks = tracker.update(
            compact_cluster(rotated_center), [0.0, 0.0, yaw], 0.2
        )
        self.assertEqual(len(tracks), 1)
        np.testing.assert_allclose(tracks[0]["world_velocity"], 0.0, atol=0.01)
        self.assertFalse(tracks[0]["urgent"])

    def test_timestamp_must_increase(self):
        tracker = LidarClusterTracker()
        points = compact_cluster([1.0, 0.0])
        tracker.update(points, [0.0, 0.0, 0.0], 1.0)
        with self.assertRaisesRegex(ValueError, "increase strictly"):
            tracker.update(points, [0.0, 0.0, 0.0], 1.0)


if __name__ == "__main__":
    unittest.main()
