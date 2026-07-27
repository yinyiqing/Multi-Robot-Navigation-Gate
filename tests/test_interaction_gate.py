import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))
sys.path.insert(0, str(ROOT / "scripts"))

from interaction_gate import InteractionGate
from robot_perception.gate_features import build_gate_feature, gate_feature_dim
from train_interaction_gate import binary_metrics, gate_metrics, select_threshold


class GateFeatureTest(unittest.TestCase):
    def test_gate_feature_has_fixed_shape(self):
        tracked = SimpleNamespace(
            local_center=np.asarray([1.0, 0.5]),
            shape_probability=0.7,
            smoothed_shape_probability=0.8,
            dynamic_speed=0.5,
            closing_speed=0.4,
            ttc=1.2,
            closest_approach_distance=0.6,
            age=3,
        )
        feature = build_gate_feature(np.zeros(24), [tracked], max_tracks=4)
        self.assertEqual(feature.shape, (gate_feature_dim(24, 4),))
        self.assertEqual(float(feature[24]), 1.0)

    def test_gate_model_output_shape(self):
        model = InteractionGate(gate_feature_dim())
        output = model(torch.zeros(3, gate_feature_dim()))
        self.assertEqual(tuple(output.shape), (3,))


class GateMetricTest(unittest.TestCase):
    def test_frame_metrics_and_threshold_selection(self):
        probabilities = np.asarray([0.9, 0.8, 0.2, 0.1])
        labels = np.asarray([1, 1, 0, 0])
        metrics = binary_metrics(probabilities, labels, 0.5)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["fpr"], 0.0)
        self.assertTrue(select_threshold(probabilities, labels)["meets_entry_criteria"])

    def test_standard_weak_fpr_is_an_entry_criterion(self):
        probabilities = np.asarray([0.9, 0.8, 0.7, 0.1])
        labels = np.asarray([1, 1, 0, 0])
        guard_mask = np.asarray([0, 0, 1, 1], dtype=bool)
        metrics = gate_metrics(probabilities, labels, 0.5, guard_mask)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["standard_weak_fpr"], 0.5)
        self.assertFalse(metrics["meets_entry_criteria"])


if __name__ == "__main__":
    unittest.main()
