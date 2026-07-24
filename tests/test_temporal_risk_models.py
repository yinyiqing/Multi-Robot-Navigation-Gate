import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from temporal_risk_models import SingleFrameRiskEncoder, TemporalRiskEncoder


class TemporalRiskModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.sequence = torch.randn(7, 8, 22)

    def test_models_produce_one_logit_per_sequence(self):
        for model in (SingleFrameRiskEncoder(), TemporalRiskEncoder()):
            self.assertEqual(tuple(model(self.sequence).shape), (7,))

    def test_single_frame_model_ignores_earlier_history(self):
        model = SingleFrameRiskEncoder()
        changed = self.sequence.clone()
        changed[:, :-1, :] += 10.0
        torch.testing.assert_close(model(self.sequence), model(changed))

    def test_temporal_model_receives_recurrent_gradients(self):
        model = TemporalRiskEncoder()
        model(self.sequence).sum().backward()
        self.assertTrue(
            all(parameter.grad is not None for parameter in model.parameters())
        )

    def test_rejects_non_sequence_input(self):
        with self.assertRaisesRegex(ValueError, "batch, time, features"):
            TemporalRiskEncoder()(torch.randn(8, 22))


if __name__ == "__main__":
    unittest.main()
