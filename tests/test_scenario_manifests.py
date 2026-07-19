import json
import gzip
import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from scenario_geometry import has_map_clearance, is_valid_map_position
from scenario_manifests import (
    PRESETS,
    generate_dataset,
    generate_scenario,
    load_manifest_dataset,
    validate_manifest_scenarios,
    write_dataset_splits,
)


class ScenarioGeometryTests(unittest.TestCase):
    def test_map_position_matches_known_free_and_blocked_points(self):
        self.assertTrue(is_valid_map_position(0.0, 0.0))
        self.assertFalse(is_valid_map_position(-2.0, 0.0))
        self.assertFalse(is_valid_map_position(5.0, 0.0))

    def test_clearance_rejects_point_near_map_boundary(self):
        self.assertTrue(is_valid_map_position(4.4, 0.0))
        self.assertFalse(has_map_clearance((4.4, 0.0), 0.24))


class ScenarioManifestTests(unittest.TestCase):
    def test_dense_generation_is_deterministic_and_valid(self):
        first = generate_scenario(1234, PRESETS["dense"])
        second = generate_scenario(1234, PRESETS["dense"])
        self.assertEqual(first, second)
        self.assertEqual(first["preset"], "dense")
        self.assertTrue(1.65 <= first["start_half_width_m"] <= 1.75)
        self.assertEqual(len(first["agents"]), 5)
        self.assertEqual(len(first["boxes"]), 4)
        self.assertGreaterEqual(first["validity"]["min_start_clearance_m"], 1.2)
        self.assertGreaterEqual(first["validity"]["min_goal_clearance_m"], 0.8)
        for agent in first["agents"].values():
            self.assertGreaterEqual(agent["task_distance_m"], 0.9)
            self.assertLessEqual(agent["task_distance_m"], 2.3)
            self.assertTrue(math.isfinite(agent["heading"]))

    def test_splits_are_fixed_and_disjoint(self):
        datasets = generate_dataset(
            PRESETS["dense"],
            {"train": 3, "validation": 2, "test": 2, "reserve": 1},
            master_seed=9,
        )
        ids = []
        for split, payload in datasets.items():
            for scenario in payload["scenarios"]:
                self.assertEqual(scenario["split"], split)
                ids.append(scenario["scenario_id"])
        self.assertEqual(len(ids), len(set(ids)))

    def test_dataset_round_trip(self):
        datasets = generate_dataset(
            PRESETS["standard"],
            {"train": 1, "validation": 0, "test": 1, "reserve": 0},
            master_seed=5,
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = write_dataset_splits(datasets, directory)
            self.assertEqual({path.name for path in paths}, {"train.json", "test.json"})
            payload = load_manifest_dataset(Path(directory) / "test.json")
            self.assertEqual(payload["split"], "test")
            json.dumps(payload)

    def test_manifest_replay_contract(self):
        scenario = generate_scenario(77, PRESETS["dense"])
        normalized = validate_manifest_scenarios([scenario], [f"r{i}" for i in range(1, 6)])
        self.assertEqual(normalized[0]["name"], scenario["scenario_id"])
        self.assertEqual(normalized[0]["layout"], "fixed")

        broken = json.loads(json.dumps(scenario))
        del broken["agents"]["r5"]
        with self.assertRaisesRegex(ValueError, "exactly match"):
            validate_manifest_scenarios([broken], [f"r{i}" for i in range(1, 6)])

    def test_loads_gzip_manifest(self):
        datasets = generate_dataset(
            PRESETS["dense"],
            {"train": 1, "validation": 0, "test": 0, "reserve": 0},
            master_seed=12,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                json.dump(datasets["train"], handle)
            payload = load_manifest_dataset(path)
            self.assertEqual(len(payload["scenarios"]), 1)

    def test_edge1_pilot_views_are_balanced_and_disjoint(self):
        view_root = (
            ROOT
            / "experiments/04_保留专门化/05_论文主线/datasets/fixed_v1/views/edge1_pilot"
        )
        train = load_manifest_dataset(view_root / "train.json.gz")["scenarios"]
        validation = load_manifest_dataset(view_root / "validation.json.gz")["scenarios"]

        self.assertEqual(len(train), 512)
        self.assertEqual(len(validation), 423)
        self.assertEqual(sum(item["preset"] == "standard" for item in train), 256)
        self.assertEqual(sum(item["preset"] == "dense" for item in train), 256)
        self.assertEqual(
            sum(item["preset"] == "standard" for item in validation), 212
        )
        self.assertEqual(sum(item["preset"] == "dense" for item in validation), 211)
        self.assertTrue(
            all(item["metrics"]["conflict_edge_count"] == 1 for item in train)
        )
        self.assertTrue(
            all(
                item["metrics"]["conflict_edge_count"] == 1
                for item in validation
            )
        )
        train_ids = {item["scenario_id"] for item in train}
        validation_ids = {item["scenario_id"] for item in validation}
        self.assertFalse(train_ids & validation_ids)


if __name__ == "__main__":
    unittest.main()
