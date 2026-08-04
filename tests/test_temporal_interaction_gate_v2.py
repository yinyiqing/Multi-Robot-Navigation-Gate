import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))
sys.path.insert(0, str(ROOT / "scripts"))

from temporal_interaction_gate import (
    TemporalInteractionGate,
    actor_comparison_features,
)
from train_temporal_interaction_gate import (
    average_precision,
    event_metrics,
    make_temporal_windows,
    project_relative_path,
    select_threshold_with_fpr_caps,
)


class ActorComparisonFeatureTest(unittest.TestCase):
    def test_actions_are_converted_to_deployed_commands(self):
        standard = np.asarray([[-1.0, 0.2], [1.0, -0.3]], dtype=np.float32)
        interaction = np.asarray([[0.0, -0.4], [-1.0, 0.1]], dtype=np.float32)
        features = actor_comparison_features(standard, interaction)
        np.testing.assert_allclose(
            features,
            np.asarray(
                [
                    [0.0, 0.2, 0.5, -0.4, 0.5, -0.6],
                    [1.0, -0.3, 0.0, 0.1, -1.0, 0.4],
                ],
                dtype=np.float32,
            ),
        )

    def test_invalid_action_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            actor_comparison_features(np.zeros((2, 3)), np.zeros((2, 3)))


class TemporalWindowTest(unittest.TestCase):
    def test_windows_do_not_cross_ego_sequence_boundaries(self):
        features = np.arange(10, dtype=np.float32).reshape(5, 2)
        sequences = [np.asarray([0, 1, 2]), np.asarray([3, 4])]
        windows, targets = make_temporal_windows(features, sequences, 3)
        np.testing.assert_array_equal(targets, [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(windows[0, -1], features[0])
        np.testing.assert_array_equal(windows[3, -1], features[3])
        self.assertTrue(np.all(windows[3, :-1] == 0.0))

    def test_temporal_model_returns_one_logit_per_window(self):
        model = TemporalInteractionGate(input_dim=7, hidden_dim=8)
        output = model(torch.zeros(4, 5, 7))
        self.assertEqual(tuple(output.shape), (4,))


class TemporalMetricTest(unittest.TestCase):
    def test_event_metrics_preserve_sequence_boundaries(self):
        labels = np.asarray([0, 1, 1, 0, 1, 0], dtype=np.uint8)
        probabilities = np.asarray([0.1, 0.8, 0.7, 0.1, 0.9, 0.1])
        sequences = [np.asarray([0, 1, 2]), np.asarray([3, 4, 5])]
        metrics = event_metrics(probabilities, labels, sequences, 0.5)
        self.assertEqual(metrics["true_events"], 2)
        self.assertEqual(metrics["predicted_events"], 2)
        self.assertEqual(metrics["event_recall"], 1.0)
        self.assertEqual(metrics["positive_interval_iou"], 1.0)

    def test_average_precision_is_one_for_perfect_ranking(self):
        self.assertEqual(
            average_precision([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]),
            1.0,
        )

    def test_project_relative_path_accepts_relative_input(self):
        self.assertEqual(
            project_relative_path("TD3/interaction_gate.py"),
            "TD3/interaction_gate.py",
        )

    def test_threshold_selection_respects_both_fpr_caps(self):
        probabilities = np.asarray([0.9, 0.8, 0.7, 0.6, 0.2, 0.1])
        labels = np.asarray([1, 0, 1, 0, 0, 0], dtype=np.uint8)
        weak_mask = np.asarray([0, 1, 0, 1, 1, 1], dtype=bool)
        metrics = select_threshold_with_fpr_caps(
            probabilities,
            labels,
            weak_mask,
            {"fpr": 0.25, "weak_fpr": 0.25},
        )
        self.assertLessEqual(metrics["fpr"], 0.25)
        self.assertLessEqual(metrics["weak_fpr"], 0.25)
        self.assertTrue(metrics["meets_fpr_caps"])


if __name__ == "__main__":
    unittest.main()
