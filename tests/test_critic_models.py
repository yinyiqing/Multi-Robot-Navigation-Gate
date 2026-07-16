import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from critic_models import Critic


class CriticModelTests(unittest.TestCase):
    def test_forward_remains_compatible_with_legacy_formula(self):
        torch.manual_seed(11)
        critic = Critic(24, 2)
        state = torch.randn(8, 24)
        action = torch.randn(8, 2)

        q1, q2 = critic(state, action)

        hidden_1 = F.relu(critic.layer_1(state))
        legacy_q1 = critic.layer_3(
            F.relu(
                torch.mm(hidden_1, critic.layer_2_s.weight.detach().t())
                + torch.mm(action, critic.layer_2_a.weight.detach().t())
                + critic.layer_2_a.bias.detach()
            )
        )
        hidden_2 = F.relu(critic.layer_4(state))
        legacy_q2 = critic.layer_6(
            F.relu(
                torch.mm(hidden_2, critic.layer_5_s.weight.detach().t())
                + torch.mm(action, critic.layer_5_a.weight.detach().t())
                + critic.layer_5_a.bias.detach()
            )
        )

        self.assertTrue(torch.allclose(q1.detach(), legacy_q1))
        self.assertTrue(torch.allclose(q2.detach(), legacy_q2))

    def test_fusion_weights_and_action_receive_gradients(self):
        critic = Critic(24, 2)
        state = torch.randn(8, 24)
        action = torch.randn(8, 2, requires_grad=True)

        q1, q2 = critic(state, action)
        (q1.mean() + q2.mean()).backward()

        required = [
            critic.layer_2_s.weight,
            critic.layer_2_a.weight,
            critic.layer_2_a.bias,
            critic.layer_5_s.weight,
            critic.layer_5_a.weight,
            critic.layer_5_a.bias,
            action,
        ]
        self.assertTrue(
            all(
                value.grad is not None and torch.isfinite(value.grad).all()
                for value in required
            )
        )


if __name__ == "__main__":
    unittest.main()
