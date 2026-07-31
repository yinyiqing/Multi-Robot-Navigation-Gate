import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from actor_objectives import (
    actor_slowdown_safety_loss,
    conservative_actor_objective,
    reference_acceleration_cap_loss,
    safe_reference_mask,
)


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

    def test_anchor_mask_only_regularizes_selected_actions(self):
        q_values = torch.tensor([[2.0], [2.0]])
        actions = torch.tensor([[1.0, 1.0], [3.0, 3.0]])
        reference = torch.zeros_like(actions)

        loss, anchor, _ = conservative_actor_objective(
            q_values,
            actions,
            reference_actions=reference,
            anchor_weight=2.0,
            anchor_mask=torch.tensor([True, False]),
        )

        self.assertAlmostEqual(anchor.item(), 1.0)
        self.assertAlmostEqual(loss.item(), 0.0)

    def test_empty_anchor_mask_keeps_q_objective(self):
        q_values = torch.tensor([[2.0], [4.0]])
        actions = torch.ones((2, 2))
        reference = torch.zeros_like(actions)

        loss, anchor, _ = conservative_actor_objective(
            q_values,
            actions,
            reference_actions=reference,
            anchor_weight=2.0,
            anchor_mask=torch.tensor([False, False]),
        )

        self.assertEqual(anchor.item(), 0.0)
        self.assertAlmostEqual(loss.item(), -3.0)

    def test_safe_reference_mask_selects_only_states_without_close_neighbors(self):
        actor_states = torch.zeros((3, 2))
        contexts = torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 2.5, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        critic_states = torch.cat((actor_states, contexts), dim=1)

        mask = safe_reference_mask(critic_states, 2, 5, 2.0)

        self.assertEqual(mask.tolist(), [False, True, True])

    def test_safe_reference_mask_rejects_malformed_context(self):
        with self.assertRaises(ValueError):
            safe_reference_mask(torch.zeros((2, 8)), 2, 5, 2.0)

    def test_slowdown_safety_loss_penalizes_fast_close_approach(self):
        actor_states = torch.zeros((3, 24))
        contexts = torch.tensor(
            [
                [0.8, 0.0, 0.8, 0.0, -0.4, 0.0, 1.0],
                [0.8, 0.0, 0.8, 0.0, 0.4, 0.0, 1.0],
                [1.4, 0.0, 1.4, 0.0, -0.4, 0.0, 1.0],
            ]
        )
        critic_states = torch.cat((actor_states, contexts), dim=1)
        actions = torch.tensor([[0.6, 0.0], [0.6, 0.0], [0.6, 0.0]])

        loss, samples = actor_slowdown_safety_loss(
            actions,
            critic_states,
            actor_state_dim=24,
            context_feature_dim=7,
            safety_distance=1.0,
            min_closing_speed=0.1,
            max_safe_linear_action=-0.4,
        )

        self.assertEqual(samples, 1)
        self.assertAlmostEqual(loss.item(), 1.0)

    def test_slowdown_safety_loss_is_zero_when_already_slow(self):
        actor_states = torch.zeros((1, 24))
        contexts = torch.tensor([[0.8, 0.0, 0.8, 0.0, -0.4, 0.0, 1.0]])
        critic_states = torch.cat((actor_states, contexts), dim=1)
        actions = torch.tensor([[-0.6, 0.0]])

        loss, samples = actor_slowdown_safety_loss(
            actions,
            critic_states,
            actor_state_dim=24,
            context_feature_dim=7,
            safety_distance=1.0,
            min_closing_speed=0.1,
            max_safe_linear_action=-0.4,
        )

        self.assertEqual(samples, 1)
        self.assertEqual(loss.item(), 0.0)

    def test_reference_cap_penalizes_only_extra_acceleration_during_approach(self):
        actor_states = torch.zeros((3, 24))
        contexts = torch.tensor(
            [
                [0.8, 0.0, 0.8, 0.0, -0.4, 0.0, 1.0],
                [0.8, 0.0, 0.8, 0.0, 0.4, 0.0, 1.0],
                [1.4, 0.0, 1.4, 0.0, -0.4, 0.0, 1.0],
            ]
        )
        critic_states = torch.cat((actor_states, contexts), dim=1)
        actor_actions = torch.tensor([[0.6, 0.0], [0.6, 0.0], [0.6, 0.0]])
        reference_actions = torch.tensor([[0.4, 0.0], [0.4, 0.0], [0.4, 0.0]])

        loss, samples = reference_acceleration_cap_loss(
            actor_actions,
            reference_actions,
            critic_states,
            actor_state_dim=24,
            context_feature_dim=7,
            safety_distance=1.0,
            min_closing_speed=0.1,
        )

        self.assertEqual(samples, 1)
        self.assertAlmostEqual(loss.item(), 0.04)

    def test_reference_cap_allows_deceleration_and_steering_changes(self):
        actor_states = torch.zeros((1, 24))
        contexts = torch.tensor([[0.8, 0.0, 0.8, 0.0, -0.4, 0.0, 1.0]])
        critic_states = torch.cat((actor_states, contexts), dim=1)

        loss, samples = reference_acceleration_cap_loss(
            torch.tensor([[0.2, 0.9]]),
            torch.tensor([[0.4, -0.9]]),
            critic_states,
            actor_state_dim=24,
            context_feature_dim=7,
            safety_distance=1.0,
            min_closing_speed=0.1,
        )

        self.assertEqual(samples, 1)
        self.assertEqual(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
