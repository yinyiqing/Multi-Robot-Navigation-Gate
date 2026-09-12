import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from g27_counterfactual import canonical_sha256
from g27_dataset import (
    _validate_features,
    audit_split,
    records_to_arrays,
    verify_payload_hash,
)


class PayloadHashTest(unittest.TestCase):
    def test_detects_mutation(self):
        payload = {"value": 1}
        payload["record_sha256"] = canonical_sha256(payload)
        self.assertEqual(
            verify_payload_hash(payload, "record_sha256"), payload["record_sha256"]
        )
        changed = copy.deepcopy(payload)
        changed["value"] = 2
        with self.assertRaisesRegex(ValueError, "record_sha256 mismatch"):
            verify_payload_hash(changed, "record_sha256")


class ArrayConversionTest(unittest.TestCase):
    def test_feature_validation_keeps_last_eight_frames_of_long_history(self):
        history = np.arange(9 * 82, dtype=np.float32).reshape(9, 82)
        reference = {
            "gate_features": {
                "sampling_stride_environment_steps": 2,
                "sequence_length": 8,
                "combined_feature_history": history.tolist(),
                "combined_feature_window": history[-8:].tolist(),
            }
        }

        window, history_length = _validate_features(reference, "anchor")

        np.testing.assert_array_equal(window, history[-8:])
        self.assertEqual(history_length, 8)

    def test_preserves_shapes_and_targets(self):
        records = [
            {
                "anchor_id": "a",
                "scenario_id": "scene-a",
                "stratum": "deep/high",
                "anchor_step": 4,
                "ego_index": 2,
                "gate_window": np.ones((8, 82), dtype=np.float32),
                "history_length": 3,
                "interaction_label": 1,
                "delta_r_1": 0.4,
                "delta_r_2": 0.2,
                "mean_delta_r": 0.3,
            }
        ]
        arrays = records_to_arrays(records)
        self.assertEqual(arrays["gate_windows"].shape, (1, 8, 82))
        self.assertEqual(arrays["interaction_labels"].tolist(), [1.0])
        self.assertEqual(arrays["history_lengths"].tolist(), [3])
        self.assertAlmostEqual(float(arrays["mean_delta_r"][0]), 0.3, places=6)

    def test_rejects_empty_dataset(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            records_to_arrays([])


class SplitAuditTest(unittest.TestCase):
    def test_audits_complete_aligned_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collection = root / "collection"
            anchor_dir = collection / "anchors/a"
            anchor_dir.mkdir(parents=True)
            anchor = {
                "anchor_id": "a",
                "scenario_id": "scene-a",
                "scenario_index": 0,
                "anchor_step": 4,
                "ego_index": 0,
                "selection_hash": "selection-a",
                "distance_stratum": "deep",
                "action_stratum": "high",
            }
            selection = {
                "format_version": 1,
                "protocol": "test",
                "selection": {
                    "selected_count": 1,
                    "cell_counts": {"deep/high": 1},
                },
                "anchors": [anchor],
            }
            selection["selection_sha256"] = canonical_sha256(selection)
            selection_path = root / "selection.json"
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            run_manifest = {
                "selection_sha256": selection["selection_sha256"],
                "anchor_ids": ["a"],
            }
            run_manifest["run_manifest_sha256"] = canonical_sha256(run_manifest)
            (collection / "run_manifest.json").write_text(
                json.dumps(run_manifest), encoding="utf-8"
            )
            reference = {
                **{key: anchor[key] for key in (
                    "anchor_id",
                    "scenario_id",
                    "scenario_index",
                    "anchor_step",
                    "ego_index",
                    "selection_hash",
                )},
                "valid": True,
                "reference_nearest_robot_distance": 1.5,
                "reference_interaction_label": 1,
                "gate_features": {
                    "sampling_stride_environment_steps": 2,
                    "sequence_length": 8,
                    "combined_feature_history": [[1.0] * 82],
                    "combined_feature_window": [[0.0] * 82] * 7 + [[1.0] * 82],
                },
            }
            reference["spec_sha256"] = canonical_sha256(reference)

            def make_branch(name, reward):
                components = {
                    "goal": 0.0,
                    "collision": 0.0,
                    "progress": reward,
                    "forward": 0.0,
                    "turn": 0.0,
                    "obstacle": 0.0,
                    "stagnation": 0.0,
                    "total": reward,
                }
                branch = {
                    "spec_sha256": reference["spec_sha256"],
                    "branch": name,
                    "actor": "interaction" if name.startswith("I") else "navigation",
                    "alignment": {"passed": True},
                    "reward": reward,
                    "reward_components": components,
                }
                branch["result_sha256"] = canonical_sha256(branch)
                return branch

            branches = {
                "N1": make_branch("N1", 1.0),
                "N2": make_branch("N2", 1.0),
                "I1": make_branch("I1", 2.0),
                "I2": make_branch("I2", 2.0),
            }
            candidate = dict(anchor)
            candidate["protocol"] = "test"
            combined = {
                "anchor_id": "a",
                "candidate": candidate,
                "reference": reference,
                "branches": branches,
                "status": "complete",
            }
            combined["record_sha256"] = canonical_sha256(combined)
            (anchor_dir / "combined.json").write_text(
                json.dumps(combined), encoding="utf-8"
            )
            summary = {
                "analysis": {
                    "completed": True,
                    "anchors_total": 1,
                    "reference_valid": 1,
                    "reference_invalid": 0,
                    "usable_anchors": 1,
                },
                "anchor_record_sha256": {"a": combined["record_sha256"]},
            }
            summary["summary_sha256"] = canonical_sha256(summary)
            (collection / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            records, audit = audit_split(selection_path, collection)
            self.assertEqual(len(records), 1)
            self.assertEqual(audit["usable"], 1)
            self.assertEqual(records[0]["mean_delta_r"], 1.0)


if __name__ == "__main__":
    unittest.main()
