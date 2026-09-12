import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))
sys.path.insert(0, str(ROOT / "scripts"))

from reward_aware_gate import (
    AuxiliaryRewardTemporalGate,
    DualEncoderRewardTemporalGate,
    MultiTaskRewardTemporalGate,
    RobustRewardAwareTemporalGate,
    RewardAwareTemporalGate,
    bounded_reward_target,
    combined_routing_score,
    conservative_routing_score,
    selective_reward_routing_score,
    auxiliary_reward_gate_loss,
    boundary_reward_tiebreak,
    joint_supervision_routing_score,
)
from train_g27_reward_aware_gate import normalize_padded_windows


class RewardAwareGateTest(unittest.TestCase):
    def test_two_heads_have_batch_shape(self):
        model = RewardAwareTemporalGate(input_dim=82, hidden_dim=8)
        interaction, advantage = model(torch.zeros(3, 8, 82))
        self.assertEqual(tuple(interaction.shape), (3,))
        self.assertEqual(tuple(advantage.shape), (3,))

    def test_advantage_adjusts_log_odds_and_is_clipped(self):
        logits = torch.zeros(3)
        advantage = torch.tensor([-4.0, 0.0, 4.0])
        scores = combined_routing_score(logits, advantage, clip=1.0)
        expected = torch.sigmoid(torch.tensor([-1.0, 0.0, 1.0]))
        self.assertTrue(torch.allclose(scores, expected))

    def test_rejects_invalid_shape(self):
        model = RewardAwareTemporalGate(input_dim=82, hidden_dim=8)
        with self.assertRaisesRegex(ValueError, "wrong shape"):
            model(torch.zeros(3, 82))

    def test_padding_remains_zero_during_normalization(self):
        windows = torch.zeros(1, 8, 82).numpy()
        windows[0, -2:] = 3.0
        normalized = normalize_padded_windows(
            windows,
            [2],
            torch.ones(82).numpy(),
            (2.0 * torch.ones(82)).numpy(),
        )
        self.assertTrue((normalized[0, :-2] == 0.0).all())
        self.assertTrue((normalized[0, -2:] == 1.0).all())

    def test_robust_gate_has_two_heads(self):
        model = RobustRewardAwareTemporalGate(input_dim=82, hidden_dim=8)
        interaction, advantage = model(torch.zeros(3, 8, 82))
        self.assertEqual(tuple(interaction.shape), (3,))
        self.assertEqual(tuple(advantage.shape), (3,))

    def test_bounded_target_clips_extreme_reward(self):
        target = bounded_reward_target(torch.tensor([-100.0, 0.0, 2.0]), 2.0)
        self.assertAlmostEqual(float(target[0]), -0.761594, places=5)
        self.assertAlmostEqual(float(target[1]), 0.0, places=7)
        self.assertAlmostEqual(float(target[2]), 0.761594, places=5)

    def test_conservative_fusion_dead_zone_preserves_zero_advantage(self):
        logits = torch.tensor([-0.3, 0.0, 2.5])
        scores = conservative_routing_score(
            logits,
            torch.tensor([0.05, 0.0, -0.8]),
            switch_on_threshold=0.43,
            fusion_alpha=0.25,
            fusion_temperature=1.0,
            advantage_dead_zone=0.1,
        )
        self.assertAlmostEqual(float(scores[0]), float(torch.sigmoid(logits[0])), places=7)
        self.assertAlmostEqual(float(scores[1]), float(torch.sigmoid(logits[1])), places=7)
        self.assertLess(float(scores[2]), float(torch.sigmoid(logits[2])))

    def test_selective_fusion_ignores_weak_reward_evidence(self):
        logits = torch.tensor([0.0, 0.0])
        scores = selective_reward_routing_score(logits, torch.tensor([0.10, 0.50]))
        self.assertAlmostEqual(float(scores[0]), 0.5, places=7)
        self.assertGreater(float(scores[1]), 0.5)

    def test_auxiliary_gate_has_single_deployable_logit(self):
        model = AuxiliaryRewardTemporalGate(82, 16)
        logits = model(torch.zeros(4, 8, 82))
        self.assertEqual(tuple(logits.shape), (4,))
        self.assertFalse(hasattr(model, "advantage_head"))

    def test_multitask_gate_returns_phase_and_reward_logits(self):
        model = MultiTaskRewardTemporalGate(82, 16)
        phase, reward = model(torch.zeros(4, 8, 82))
        self.assertEqual(tuple(phase.shape), (4,))
        self.assertEqual(tuple(reward.shape), (4,))

    def test_auxiliary_loss_masks_noisy_reward_differences(self):
        logits = torch.zeros(4)
        labels = torch.tensor([0., 1., 0., 1.])
        delta = torch.tensor([0.001, 0.02, -0.03, 0.0])
        total, primary, auxiliary, mask = auxiliary_reward_gate_loss(
            logits, labels, delta, noise_threshold=0.005, reward_weight=0.25
        )
        self.assertEqual(mask.tolist(), [False, True, True, False])
        self.assertGreater(float(total), 0.0)
        self.assertGreater(float(auxiliary), 0.0)

    def test_reward_tiebreak_only_applies_in_uncertain_band(self):
        p = torch.tensor([0.2, 0.5, 0.8, 0.5])
        delta = torch.tensor([0.1, 0.1, -0.1, 0.001])
        out = boundary_reward_tiebreak(p, delta)
        self.assertTrue(torch.allclose(out, torch.tensor([0.2, 1.0, 0.8, 0.5])))

    def test_dual_encoder_has_independent_temporal_parameters(self):
        model = DualEncoderRewardTemporalGate(82, 16)
        interaction, reward = model(torch.zeros(4, 8, 82))
        self.assertEqual(tuple(interaction.shape), (4,))
        self.assertEqual(tuple(reward.shape), (4,))
        self.assertIsNot(model.interaction_gru, model.reward_gru)
        self.assertTrue(torch.all(reward >= -1.0))
        self.assertTrue(torch.all(reward <= 1.0))

    def test_joint_score_uses_both_predictions_with_bounded_effect(self):
        logits = torch.zeros(3)
        scores = joint_supervision_routing_score(
            logits, torch.tensor([-2.0, 0.0, 2.0]), reward_weight=0.25
        )
        expected = torch.sigmoid(torch.tensor([-0.25, 0.0, 0.25]))
        self.assertTrue(torch.allclose(scores, expected))


if __name__ == "__main__":
    unittest.main()
