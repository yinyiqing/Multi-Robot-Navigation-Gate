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
from train_g11_b_aggregated_gate import (
    concatenate_sources,
    group_balance_weights,
    normalize_source_balanced,
    source_scenario_sample_weights,
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


class AggregatedDatasetTest(unittest.TestCase):
    @staticmethod
    def dataset(prefix, lengths):
        frame_count = sum(lengths)
        sequences = []
        offset = 0
        scenarios = []
        for index, length in enumerate(lengths):
            sequences.append(np.arange(offset, offset + length))
            scenarios.extend(["%s-%d" % (prefix, index)] * length)
            offset += length
        from train_temporal_interaction_gate import GateDataset

        return GateDataset(
            base_features=np.arange(frame_count * 2, dtype=np.float32).reshape(
                frame_count, 2
            ),
            actor_features=np.zeros((frame_count, 1), dtype=np.float32),
            labels={
                "front": np.zeros(frame_count, dtype=np.float32),
                "any": np.zeros(frame_count, dtype=np.float32),
            },
            scenarios=np.asarray(scenarios),
            strata=np.asarray(["standard_weak"] * frame_count),
            sequence_indices=sequences,
        )

    def test_concatenation_offsets_sequences_and_prefixes_sources(self):
        first = self.dataset("case", [2, 1])
        second = self.dataset("case", [3])
        result = concatenate_sources((("a1", first), ("student", second)))
        self.assertEqual(len(result.base_features), 6)
        np.testing.assert_array_equal(result.sequence_indices[-1], [3, 4, 5])
        self.assertEqual(result.scenarios[0], "a1::case-0")
        self.assertEqual(result.scenarios[-1], "student::case-0")

    def test_group_balance_gives_every_source_scenario_equal_mass(self):
        groups = np.asarray(["a", "a", "b", "b", "b", "c"])
        weights = group_balance_weights(groups)
        masses = [float(np.sum(weights[groups == name])) for name in ("a", "b", "c")]
        np.testing.assert_allclose(masses, [1.0 / 3.0] * 3)

    def test_training_weights_keep_groups_equal_after_class_balance(self):
        groups = np.asarray(["long", "long", "long", "short", "short"])
        labels = np.asarray([0, 0, 1, 0, 1])
        weights = source_scenario_sample_weights(labels, groups)
        self.assertAlmostEqual(
            float(np.sum(weights[groups == "long"])),
            float(np.sum(weights[groups == "short"])),
        )
        self.assertAlmostEqual(float(weights[0] + weights[1]), float(weights[2]))
        self.assertAlmostEqual(float(weights[3]), float(weights[4]))

    def test_source_balanced_normalization_uses_group_not_frame_mean(self):
        train = np.asarray([[0.0], [0.0], [0.0], [10.0]], dtype=np.float32)
        groups = np.asarray(["long", "long", "long", "short"])
        normalized, validation, mean, std = normalize_source_balanced(
            train, np.asarray([[5.0]], dtype=np.float32), groups
        )
        self.assertAlmostEqual(float(mean[0]), 5.0)
        self.assertAlmostEqual(float(std[0]), 5.0)
        self.assertAlmostEqual(float(np.mean(normalized[:3, 0])), -1.0)
        self.assertAlmostEqual(float(normalized[-1, 0]), 1.0)
        self.assertAlmostEqual(float(validation[0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
