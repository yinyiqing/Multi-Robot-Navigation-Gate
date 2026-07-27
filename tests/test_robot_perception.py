import math
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_robot_perception_views import split_sizes
from robot_perception.dataset import build_frame_examples, load_shard
from robot_perception.metrics import detection_metrics
from robot_perception.models import LocalRobotDetector
from robot_perception.range_view import extract_local_patch, project_range_view
from robot_perception.recorder import PerceptionShardRecorder


def synthetic_points():
    robot = np.asarray(
        [
            [0.62, -0.05, 0.00],
            [0.64, 0.00, 0.03],
            [0.66, 0.05, 0.06],
            [0.68, 0.08, 0.02],
        ],
        dtype=np.float32,
    )
    background = np.asarray(
        [
            [1.90, 0.92, 0.00],
            [1.94, 0.96, 0.04],
            [1.98, 1.00, 0.08],
            [2.02, 1.04, 0.02],
        ],
        dtype=np.float32,
    )
    return np.concatenate((robot, background), axis=0)


class RangeViewTest(unittest.TestCase):
    def test_projection_keeps_nearest_return(self):
        points = np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
        view = project_range_view(points)
        self.assertEqual(int(view.valid.sum()), 1)
        self.assertAlmostEqual(float(view.ranges[view.valid][0]), 1.0, places=5)

    def test_fixed_width_patch_has_stable_shape(self):
        view = project_range_view(synthetic_points())
        patch = extract_local_patch(view, np.asarray([0.65, 0.0]))
        self.assertEqual(patch.shape, (3, 16, 64))
        self.assertGreater(float(patch[2].sum()), 0.0)
        self.assertTrue(np.all(np.abs(patch[:2]) <= 1.0))


class DatasetTest(unittest.TestCase):
    def test_candidate_matching_separates_robot_and_background(self):
        examples = build_frame_examples(
            synthetic_points(),
            ego_pose=np.asarray([0.0, 0.0, 0.0]),
            other_world_positions=[np.asarray([1.0, 0.0])],
        )
        self.assertEqual(examples.visible_robot_count, 1)
        self.assertEqual(examples.missed_visible_robot_count, 0)
        self.assertEqual(int(examples.labels.sum()), 1)
        self.assertIn(0, examples.labels.tolist())

    def test_metrics_recall_includes_missed_proposals(self):
        metrics = detection_metrics(
            np.asarray([0.9, 0.1]),
            np.asarray([1, 0]),
            visible_robot_count=2,
            threshold=0.5,
        )
        self.assertEqual(metrics.precision, 1.0)
        self.assertEqual(metrics.recall, 0.5)
        self.assertEqual(metrics.proposal_recall, 0.5)
        self.assertEqual(metrics.missed_visible_robots, 1)

    def test_shard_recorder_is_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = PerceptionShardRecorder(directory, "train", frame_stride=1)
            self.assertTrue(recorder.begin_scenario("case/one", "standard", "weak"))
            recorder.record_frame(
                {"r1": synthetic_points(), "r2": np.empty((0, 3))},
                {"r1": [0.0, 0.0, 0.0], "r2": [1.0, 0.0, math.pi]},
                [True, True],
                ["r1", "r2"],
            )
            result = recorder.finish_scenario()
            self.assertTrue(result["written"])
            shard = load_shard(Path(directory) / "case_one.npz")
            self.assertEqual(int(shard["visible_robot_count"]), 1)
            self.assertEqual(str(shard["scenario_pool"]), "standard")
            self.assertEqual(str(shard["interaction_band"]), "weak")
            self.assertFalse(recorder.begin_scenario("case/one"))
            result = recorder.finish_scenario()
            self.assertFalse(result["written"])


class ModelAndSplitTest(unittest.TestCase):
    def test_detector_output_shapes(self):
        model = LocalRobotDetector()
        logits, offsets = model(torch.zeros(4, 3, 16, 64))
        self.assertEqual(tuple(logits.shape), (4,))
        self.assertEqual(tuple(offsets.shape), (4, 2))

    def test_largest_remainder_split(self):
        self.assertEqual(split_sizes(1159, (0.8, 0.1, 0.1)), (927, 116, 116))
        self.assertEqual(split_sizes(1841, (0.8, 0.1, 0.1)), (1473, 184, 184))

    def test_fixed_perception_views_are_disjoint(self):
        root = (
            ROOT
            / "experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/robot_perception_v1"
        )
        expected = {"train": 7200, "validation": 900, "test": 900}
        split_ids = {}
        for split, count in expected.items():
            with gzip.open(root / f"{split}.json.gz", "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(len(payload["scenarios"]), count)
            split_ids[split] = {
                scenario["scenario_id"] for scenario in payload["scenarios"]
            }
            self.assertTrue(
                all(
                    scenario["split"] == split
                    and scenario["navigation_split"] == "train"
                    for scenario in payload["scenarios"]
                )
            )
        self.assertFalse(split_ids["train"] & split_ids["validation"])
        self.assertFalse(split_ids["train"] & split_ids["test"])
        self.assertFalse(split_ids["validation"] & split_ids["test"])

    def test_perception_pilot_views_are_balanced(self):
        root = (
            ROOT
            / "experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/robot_perception_v1"
        )
        for split in ("train", "validation"):
            with gzip.open(
                root / f"pilot_{split}.json.gz", "rt", encoding="utf-8"
            ) as handle:
                scenarios = json.load(handle)["scenarios"]
            self.assertEqual(len(scenarios), 100)
            for pool in ("standard", "dense"):
                for band in ("weak", "interaction"):
                    self.assertEqual(
                        sum(
                            scenario["view"]["perception_pool"] == pool
                            and scenario["view"]["interaction_band"] == band
                            for scenario in scenarios
                        ),
                        25,
                    )


if __name__ == "__main__":
    unittest.main()
