import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from actor_objectives import conservative_actor_objective


class ActorObjectiveTests(unittest.TestCase):
    def test_legacy_objective_is_negative_mean_q(self):
        q_values = torch.tensor([[2.0], [4.0]])
        actions = torch.zeros((2, 2))
        loss, anchor, scale = conservative_actor_objective(q_values, actions)
        self.assertAlmostEqual(loss.item(), -3.0)
        self.assertEqual(anchor.item(), 0.0)
        self.assertEqual(scale.item(), 1.0)

    def test_normalized_q_is_invariant_to_q_scale(self):
        actions = torch.zeros((2, 2))
        first, _, _ = conservative_actor_objective(
            torch.tensor([[2.0], [4.0]]),
            actions,
            q_normalization_alpha=1.0,
        )
        second, _, _ = conservative_actor_objective(
            torch.tensor([[20.0], [40.0]]),
            actions,
            q_normalization_alpha=1.0,
        )
        self.assertAlmostEqual(first.item(), second.item())

    def test_anchor_penalizes_departure_from_reference(self):
        q_values = torch.tensor([[1.0]])
        actions = torch.tensor([[0.2, -0.2]])
        reference = torch.zeros((1, 2))
        loss, anchor, _ = conservative_actor_objective(
            q_values,
            actions,
            reference,
            q_normalization_alpha=1.0,
            anchor_weight=2.5,
        )
        self.assertAlmostEqual(anchor.item(), 0.04)
        self.assertAlmostEqual(loss.item(), -0.9)

    def test_rejects_negative_weights(self):
        q_values = torch.ones((1, 1))
        actions = torch.zeros((1, 2))
        with self.assertRaises(ValueError):
            conservative_actor_objective(
                q_values, actions, q_normalization_alpha=-1.0
            )
        with self.assertRaises(ValueError):
            conservative_actor_objective(q_values, actions, anchor_weight=-1.0)


if __name__ == "__main__":
    unittest.main()
