import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "TD3"))

from residual_teacher_audit import (
    VectorScaledResidual,
    balanced_class_weights,
    calibrate_residual_scale,
    interaction_labels_from_critic_states,
    teacher_choice_accuracy,
)


class ResidualTeacherAuditTests(unittest.TestCase):
    def test_interaction_labels_use_nearest_valid_neighbor(self):
        actor = np.zeros((3, 2), dtype=np.float32)
        context = np.array(
            [
                [0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        critic = np.concatenate([actor, context], axis=1)
        labels = interaction_labels_from_critic_states(critic, 2, 7, 2.0)
        np.testing.assert_array_equal(labels, [True, False, False])

    def test_scale_is_calibrated_per_action(self):
        deltas = np.array([[0.1, 0.4], [-0.2, 0.8]], dtype=np.float32)
        scale = calibrate_residual_scale(
            deltas, quantile=1.0, minimum=0.05, maximum=1.0
        )
        np.testing.assert_allclose(scale, [0.2, 0.8])

    def test_default_scale_can_cover_full_actor_action_difference(self):
        deltas = np.array([[-1.8, 1.6], [-1.9, 1.7]], dtype=np.float32)
        scale = calibrate_residual_scale(deltas, quantile=1.0)
        np.testing.assert_allclose(scale, [1.9, 1.7])

    def test_class_weights_balance_both_classes(self):
        labels = np.array([False, False, False, True])
        weights = balanced_class_weights(labels)
        self.assertAlmostEqual(float(np.sum(weights[labels])), 2.0)
        self.assertAlmostEqual(float(np.sum(weights[~labels])), 2.0)

    def test_vector_scaled_residual_respects_each_bound(self):
        model = VectorScaledResidual(3, 2, 4, [0.1, 0.3])
        with torch.no_grad():
            model.layer_2.weight.fill_(100.0)
            model.layer_2.bias.fill_(100.0)
        output = model(torch.ones((2, 3)))
        self.assertTrue(torch.all(output[:, 0] <= 0.1))
        self.assertTrue(torch.all(output[:, 1] <= 0.3))

    def test_teacher_choice_is_reported_by_class(self):
        generalist = np.zeros((2, 2), dtype=np.float32)
        specialist = np.ones((2, 2), dtype=np.float32)
        labels = np.array([False, True])
        residual = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        result = teacher_choice_accuracy(
            residual, generalist, specialist, labels
        )
        self.assertEqual(result["normal"]["accuracy"], 1.0)
        self.assertEqual(result["interaction"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
