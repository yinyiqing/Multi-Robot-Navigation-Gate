import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from actor_models import Actor
from interaction_expert import StrongInteractionTD3, TemporalStrongActor
from sequence_replay_buffer import SequenceReplayBuffer


class TemporalStrongActorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.base = Actor(24, 2)
        self.histories = torch.randn(6, 8, 24)

    def test_zero_initialized_actor_exactly_matches_5d_warm_start(self):
        actor = TemporalStrongActor(self.base.state_dict())
        expected = self.base(self.histories[:, -1])
        actual, base_action, residual = actor(self.histories, return_details=True)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(base_action, expected))
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))

    def test_warm_started_backbone_and_temporal_head_are_trainable(self):
        actor = TemporalStrongActor(self.base.state_dict())
        actor(self.histories).square().mean().backward()
        self.assertTrue(all(p.requires_grad for p in actor.backbone.parameters()))
        self.assertIsNotNone(actor.backbone.layer_1.weight.grad)
        self.assertIsNotNone(actor.residual_head[2].weight.grad)

    def test_history_shape_is_checked(self):
        actor = TemporalStrongActor(self.base.state_dict())
        with self.assertRaises(ValueError):
            actor(torch.randn(2, 7, 24))


class StrongInteractionTD3Tests(unittest.TestCase):
    def test_training_step_updates_critic_before_actor_unlock(self):
        torch.manual_seed(3)
        np.random.seed(3)
        base = Actor(24, 2)
        agent = StrongInteractionTD3(base.state_dict(), hidden_dim=32, device="cpu")
        replay = SequenceReplayBuffer(
            100, group_ratios={"deep": 0.6, "close": 0.2, "margin": 0.2}
        )
        for index in range(30):
            history = np.random.randn(8, 24).astype(np.float32)
            replay.add(
                history,
                np.zeros(2, dtype=np.float32),
                0.0,
                False,
                history,
                ("deep", "close", "margin")[index % 3],
            )
        metrics = agent.train_step(replay, batch_size=12, update_actor=False)
        self.assertIn("critic_loss", metrics)
        self.assertNotIn("actor_loss", metrics)


if __name__ == "__main__":
    unittest.main()
