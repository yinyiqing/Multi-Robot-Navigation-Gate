import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from actor_models import (
    Actor,
    ResidualActor,
    actor_hidden_dims_from_state_dict,
    function_preserving_expand_actor_state_dict,
    is_residual_actor_state_dict,
)


class ActorModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.state_dim = 24
        self.action_dim = 2
        self.base_actor = Actor(self.state_dim, self.action_dim)
        self.states = torch.randn(16, self.state_dim)

    def test_actor_checkpoint_keys_remain_compatible(self):
        self.assertEqual(
            set(self.base_actor.state_dict()),
            {
                "layer_1.weight",
                "layer_1.bias",
                "layer_2.weight",
                "layer_2.bias",
                "layer_3.weight",
                "layer_3.bias",
            },
        )

    def test_capacity_matched_actor_parameter_count(self):
        wide_actor = Actor(
            self.state_dim,
            self.action_dim,
            hidden_dim_1=1137,
            hidden_dim_2=855,
        )
        parameter_count = sum(parameter.numel() for parameter in wide_actor.parameters())

        self.assertEqual(parameter_count, 1_003_127)
        self.assertEqual(
            actor_hidden_dims_from_state_dict(wide_actor.state_dict()), (1137, 855)
        )

    def test_function_preserving_expansion_matches_base_and_keeps_new_branch_live(self):
        wide_actor = Actor(
            self.state_dim,
            self.action_dim,
            hidden_dim_1=1137,
            hidden_dim_2=855,
        )
        expanded = function_preserving_expand_actor_state_dict(
            self.base_actor.state_dict(), wide_actor
        )
        wide_actor.load_state_dict(expanded)

        expected = self.base_actor(self.states)
        actual = wide_actor(self.states)
        self.assertLessEqual((actual - expected).abs().max().item(), 1e-6)

        loss = wide_actor(self.states).square().mean()
        loss.backward()
        added_output_gradient = wide_actor.layer_3.weight.grad[:, 600:]
        self.assertGreater(added_output_gradient.abs().sum().item(), 0.0)

    def test_expansion_rejects_narrower_target(self):
        narrow_actor = Actor(
            self.state_dim,
            self.action_dim,
            hidden_dim_1=799,
            hidden_dim_2=599,
        )
        with self.assertRaisesRegex(ValueError, "must not be narrower"):
            function_preserving_expand_actor_state_dict(
                self.base_actor.state_dict(), narrow_actor
            )

    def test_zero_initialized_residual_matches_base_actor(self):
        residual_actor = ResidualActor(
            self.state_dim, self.action_dim, hidden_dim=64, residual_scale=0.15
        )
        residual_actor.load_base_state_dict(self.base_actor.state_dict())

        expected = self.base_actor(self.states)
        actual = residual_actor(self.states)

        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(residual_actor.residual(self.states), torch.zeros_like(expected)))

    def test_only_residual_parameters_receive_gradients(self):
        residual_actor = ResidualActor(self.state_dim, self.action_dim)
        residual_actor.load_base_state_dict(self.base_actor.state_dict())

        loss = residual_actor(self.states).square().mean()
        loss.backward()

        self.assertTrue(
            all(parameter.grad is None for parameter in residual_actor.base_actor.parameters())
        )
        self.assertIsNotNone(residual_actor.adapter_layer_2.weight.grad)
        self.assertGreater(residual_actor.adapter_layer_2.weight.grad.abs().sum().item(), 0.0)

    def test_residual_checkpoint_round_trip_preserves_output_and_scale(self):
        residual_actor = ResidualActor(
            self.state_dim, self.action_dim, hidden_dim=64, residual_scale=0.2
        )
        residual_actor.load_base_state_dict(self.base_actor.state_dict())
        with torch.no_grad():
            residual_actor.adapter_layer_2.weight.normal_(0.0, 0.01)

        clone = ResidualActor(
            self.state_dim, self.action_dim, hidden_dim=64, residual_scale=0.05
        )
        clone.load_state_dict(residual_actor.state_dict())

        self.assertTrue(is_residual_actor_state_dict(clone.state_dict()))
        self.assertAlmostEqual(clone.residual_scale, 0.2)
        self.assertTrue(torch.equal(clone(self.states), residual_actor(self.states)))


if __name__ == "__main__":
    unittest.main()
