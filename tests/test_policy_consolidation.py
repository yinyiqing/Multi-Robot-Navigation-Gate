import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from policy_consolidation import (
    consolidation_metrics,
    initialize_augmented_actor,
    load_consolidation_dataset,
    scenario_class_weights,
    select_teacher_actions,
)
from actor_models import Actor


class PolicyConsolidationTest(unittest.TestCase):
    def test_teacher_selection_uses_generalist_then_specialist(self):
        generalist = np.asarray([[0.8, 0.1], [0.7, -0.2]], dtype=np.float32)
        specialist = np.asarray([[0.2, 0.4], [0.1, -0.5]], dtype=np.float32)
        selected = select_teacher_actions(generalist, specialist, [0, 1])
        np.testing.assert_allclose(selected, [[0.8, 0.1], [0.1, -0.5]])

    def test_scenario_class_weights_balance_classes_and_scenarios(self):
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        scenarios = np.asarray(["a", "a", "b", "c", "d", "d"])
        weights = scenario_class_weights(labels, scenarios)
        self.assertAlmostEqual(float(weights[labels == 0].sum()), 3.0, places=5)
        self.assertAlmostEqual(float(weights[labels == 1].sum()), 3.0, places=5)
        self.assertAlmostEqual(float(weights[:2].sum()), float(weights[2]), places=5)
        self.assertAlmostEqual(float(weights[3]), float(weights[4:].sum()), places=5)

    def test_perfect_student_has_zero_error_and_correct_teacher_choice(self):
        generalist = np.asarray([[0.9, 0.0], [0.8, 0.1]], dtype=np.float32)
        specialist = np.asarray([[0.1, 0.4], [0.0, -0.3]], dtype=np.float32)
        labels = np.asarray([0, 1])
        student = select_teacher_actions(generalist, specialist, labels)
        metrics = consolidation_metrics(
            student, generalist, specialist, labels, strata=["normal", "strong"]
        )
        self.assertEqual(metrics["all"]["mse"], 0.0)
        self.assertEqual(metrics["teacher_choice_accuracy"], 1.0)
        self.assertEqual(set(metrics["strata"]), {"normal", "strong"})

    def test_augmented_actor_starts_as_exact_source_actor(self):
        torch.manual_seed(3)
        source = Actor(24, 2)
        student = Actor(76, 2)
        initialize_augmented_actor(student, source, 24)
        actor_states = torch.randn(8, 24)
        augmented = torch.cat((actor_states, torch.randn(8, 52)), dim=1)
        self.assertTrue(torch.equal(student(augmented), source(actor_states)))

    def test_dataset_loader_checks_fields_and_preserves_scenario_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.npz"
            np.savez_compressed(
                path,
                frame_actor_states=np.zeros((2, 24), dtype=np.float32),
                frame_oracle_interaction_labels=np.asarray([0, 1], dtype=np.uint8),
                frame_front_interaction_labels=np.asarray([0, 1], dtype=np.uint8),
                scenario_id=np.asarray("case-1"),
                scenario_pool=np.asarray("standard"),
                interaction_band=np.asarray("weak"),
            )
            dataset = load_consolidation_dataset([path])
            self.assertEqual(dataset["states"].shape, (2, 24))
            self.assertEqual(dataset["scenarios"].tolist(), ["case-1", "case-1"])
            self.assertEqual(dataset["strata"].tolist(), ["standard_weak"] * 2)


if __name__ == "__main__":
    unittest.main()
