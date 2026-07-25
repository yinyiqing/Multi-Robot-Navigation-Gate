import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from critic_safety_ranking import (
    approaching_safety_mask,
    critic_safety_ranking_loss,
)


class LinearActionCritic(nn.Module):
    def __init__(self, direction):
        super().__init__()
        self.direction = nn.Parameter(torch.tensor(float(direction)))

    def forward(self, state, action):
        value = self.direction * action[:, :1]
        return value, value


def critic_state(distance, relative_x_velocity):
    actor_state = [0.0] * 24
    context = [distance, 0.0, distance, 0.0, relative_x_velocity, 0.0, 1.0]
    return actor_state + context


class CriticSafetyRankingTests(unittest.TestCase):
    def test_mask_selects_only_close_approaching_neighbors(self):
        states = torch.tensor(
            [
                critic_state(0.8, -0.4),
                critic_state(0.8, 0.4),
                critic_state(1.4, -0.4),
            ]
        )
        mask = approaching_safety_mask(states, 24, 7, 1.0, 0.1)
        self.assertEqual(mask.tolist(), [True, False, False])

    def test_ranking_penalizes_critic_that_prefers_faster_action(self):
        critic = LinearActionCritic(1.0)
        states = torch.tensor([critic_state(0.8, -0.4)])
        actions = torch.tensor([[0.8, 0.0]])
        loss, samples = critic_safety_ranking_loss(
            critic, states, actions, 24, 7, 1.0, 0.1, 0.4, 0.1
        )
        self.assertEqual(samples, 1)
        self.assertAlmostEqual(float(loss), 1.0, places=6)

    def test_ranking_accepts_critic_that_prefers_slower_action(self):
        critic = LinearActionCritic(-1.0)
        states = torch.tensor([critic_state(0.8, -0.4)])
        actions = torch.tensor([[0.8, 0.0]])
        loss, samples = critic_safety_ranking_loss(
            critic, states, actions, 24, 7, 1.0, 0.1, 0.4, 0.1
        )
        self.assertEqual(samples, 1)
        self.assertEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
