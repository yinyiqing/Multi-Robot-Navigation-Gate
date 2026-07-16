import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from actor_models import Actor, ResidualActor, is_residual_actor_state_dict


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
