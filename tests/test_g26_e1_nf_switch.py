import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))
sys.path.insert(0, str(ROOT / "scripts"))

from nf_switch_controller import (
    NormalizingFlowActorSwitcher,
    empirical_cdf_score,
)
from normalizing_flow import RealNVPFlow, alternating_binary_masks
from train_g26_e1_nf_switch import (
    normalize_fit_calibration,
    scenario_equal_frame_weights,
    split_success_scenarios,
)


class FixedPolicy:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)

    def get_action(self, state):
        del state
        return self.action.copy()


class RealNVPFlowTest(unittest.TestCase):
    def test_realnvp_is_invertible_and_returns_per_frame_nll(self):
        torch.manual_seed(20260821)
        flow = RealNVPFlow(
            input_dim=6,
            num_blocks=4,
            hidden_dim=16,
            log_scale_limit=1.0,
        )
        values = torch.randn(11, 6)

        latent, forward_log_det = flow(values)
        restored, inverse_log_det = flow.inverse(latent)
        nll = flow.negative_log_likelihood(values)

        self.assertEqual(tuple(latent.shape), (11, 6))
        self.assertEqual(tuple(nll.shape), (11,))
        self.assertTrue(torch.all(torch.isfinite(nll)))
        torch.testing.assert_close(restored, values, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            forward_log_det + inverse_log_det,
            torch.zeros_like(forward_log_det),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_realnvp_masks_alternate_transformed_dimensions(self):
        masks = alternating_binary_masks(4, 3)
        self.assertEqual(
            [mask.tolist() for mask in masks],
            [
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ],
        )


class NormalizingFlowSwitcherTest(unittest.TestCase):
    def test_nf_switch_uses_calibration_threshold_and_records_scores(self):
        model_config = {
            "input_dim": 24,
            "num_blocks": 2,
            "hidden_dim": 8,
            "log_scale_limit": 1.0,
        }
        flow = RealNVPFlow(**model_config)
        for parameter in flow.parameters():
            torch.nn.init.zeros_(parameter)

        zero_state_nll = 0.5 * 24 * math.log(2.0 * math.pi)
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "checkpoint.pt"
            torch.save(
                {
                    "model_config": model_config,
                    "model_state_dict": flow.state_dict(),
                    "feature_mean": np.zeros(24, dtype=np.float32),
                    "feature_std": np.ones(24, dtype=np.float32),
                    "threshold_nll": zero_state_nll + 1.0,
                    "calibration_nll_sorted": np.asarray(
                        [zero_state_nll, zero_state_nll + 1.0, zero_state_nll + 5.0],
                        dtype=np.float32,
                    ),
                },
                checkpoint_path,
            )

            switcher = NormalizingFlowActorSwitcher(
                standard_policy=FixedPolicy([0.1, 0.2]),
                interaction_policy=FixedPolicy([0.3, 0.4]),
                flow_checkpoint=checkpoint_path,
                device="cpu",
            )
            switcher.reset(["r1"])

            low_action, low_mode, low_score, low_diag = switcher.choose_action(
                None, "r1", np.zeros(24, dtype=np.float32)
            )
            high_action, high_mode, high_score, high_diag = switcher.choose_action(
                None, "r1", np.ones(24, dtype=np.float32)
            )

        self.assertEqual(low_mode, "standard")
        self.assertEqual(high_mode, "dense")
        np.testing.assert_allclose(low_action, [0.1, 0.2])
        np.testing.assert_allclose(high_action, [0.3, 0.4])
        self.assertLess(low_diag["nll"], high_diag["nll"])
        self.assertEqual(
            low_score,
            empirical_cdf_score(low_diag["nll"], switcher.calibration_nll_sorted),
        )
        self.assertEqual(
            high_score,
            empirical_cdf_score(high_diag["nll"], switcher.calibration_nll_sorted),
        )
        self.assertEqual(switcher.episode_stats()["switches"], 1)


class G26E1TrainingHelperTest(unittest.TestCase):
    def test_e1_split_keeps_only_full_success_by_registered_strata(self):
        scenarios = {
            "dense_a": {"view": {"gate_pool": "dense", "interaction_band": "weak"}},
            "dense_b": {"view": {"gate_pool": "dense", "interaction_band": "weak"}},
            "std_a": {
                "view": {"gate_pool": "standard", "interaction_band": "interaction"}
            },
            "std_b": {
                "view": {"gate_pool": "standard", "interaction_band": "interaction"}
            },
            "std_c": {
                "view": {"gate_pool": "standard", "interaction_band": "interaction"}
            },
        }
        outcomes = {
            "dense_a": {"full_success": 1},
            "dense_b": {"full_success": 0},
            "std_a": {"full_success": 1},
            "std_b": {"full_success": 1},
            "std_c": {"full_success": 1},
        }

        fit_ids, calibration_ids, strata = split_success_scenarios(
            scenarios,
            outcomes,
            split_seed="20260821",
            fit_fraction=0.80,
        )

        self.assertEqual(
            set(fit_ids + calibration_ids),
            {"dense_a", "std_a", "std_b", "std_c"},
        )
        self.assertNotIn("dense_b", fit_ids + calibration_ids)
        self.assertEqual(
            strata["dense_weak"],
            {
                "success_scenarios": 1,
                "fit_scenarios": 0,
                "calibration_scenarios": 1,
            },
        )
        self.assertEqual(
            strata["standard_interaction"],
            {
                "success_scenarios": 3,
                "fit_scenarios": 2,
                "calibration_scenarios": 1,
            },
        )

    def test_training_helpers_use_fit_only_statistics_and_equal_scenario_mass(self):
        fit = np.asarray([[0.0, 2.0], [2.0, 4.0]], dtype=np.float32)
        calibration = np.asarray([[10.0, 20.0]], dtype=np.float32)
        fit_values, calibration_values, mean, std, floored = normalize_fit_calibration(
            fit, calibration
        )

        np.testing.assert_allclose(mean, [1.0, 3.0])
        np.testing.assert_allclose(std, [1.0, 1.0])
        np.testing.assert_allclose(fit_values, [[-1.0, -1.0], [1.0, 1.0]])
        np.testing.assert_allclose(calibration_values, [[9.0, 17.0]])
        self.assertFalse(np.any(floored))

        groups = np.asarray(["long", "long", "short"])
        weights = scenario_equal_frame_weights(groups)
        self.assertTrue(
            np.isclose(
                np.sum(weights[groups == "long"]),
                np.sum(weights[groups == "short"]),
            )
        )


if __name__ == "__main__":
    unittest.main()
