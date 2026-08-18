import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))
sys.path.insert(0, str(ROOT / "scripts"))

from interaction_gate import InteractionGate
from learned_gate_controller import (
    GateHysteresis,
    LearnedInteractionGateController,
    infer_max_tracks,
)
from robot_perception.gate_features import build_gate_feature, gate_feature_dim
from robot_perception.models import LocalRobotDetector
from temporal_interaction_gate import TemporalInteractionGate
from train_interaction_gate import binary_metrics, gate_metrics, select_threshold
from train_temporal_interaction_gate import make_temporal_windows


class GateFeatureTest(unittest.TestCase):
    def test_gate_feature_has_fixed_shape(self):
        tracked = SimpleNamespace(
            local_center=np.asarray([1.0, 0.5]),
            shape_probability=0.7,
            smoothed_shape_probability=0.8,
            dynamic_speed=0.5,
            closing_speed=0.4,
            ttc=1.2,
            closest_approach_distance=0.6,
            age=3,
        )
        feature = build_gate_feature(np.zeros(24), [tracked], max_tracks=4)
        self.assertEqual(feature.shape, (gate_feature_dim(24, 4),))
        self.assertEqual(float(feature[24]), 1.0)

    def test_gate_model_output_shape(self):
        model = InteractionGate(gate_feature_dim())
        output = model(torch.zeros(3, gate_feature_dim()))
        self.assertEqual(tuple(output.shape), (3,))

    def test_single_frame_temporal_windows_contain_no_history(self):
        features = np.arange(12, dtype=np.float32).reshape(4, 3)
        windows, indices = make_temporal_windows(
            features, [np.asarray([0, 1]), np.asarray([2, 3])], 1
        )
        self.assertEqual(windows.shape, (4, 1, 3))
        np.testing.assert_array_equal(windows[:, 0], features)
        np.testing.assert_array_equal(indices, np.arange(4))


class GateMetricTest(unittest.TestCase):
    def test_frame_metrics_and_threshold_selection(self):
        probabilities = np.asarray([0.9, 0.8, 0.2, 0.1])
        labels = np.asarray([1, 1, 0, 0])
        metrics = binary_metrics(probabilities, labels, 0.5)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["fpr"], 0.0)
        self.assertTrue(select_threshold(probabilities, labels)["meets_entry_criteria"])

    def test_standard_weak_fpr_is_an_entry_criterion(self):
        probabilities = np.asarray([0.9, 0.8, 0.7, 0.1])
        labels = np.asarray([1, 1, 0, 0])
        guard_mask = np.asarray([0, 0, 1, 1], dtype=bool)
        metrics = gate_metrics(probabilities, labels, 0.5, guard_mask)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["standard_weak_fpr"], 0.5)
        self.assertFalse(metrics["meets_entry_criteria"])


class GateHysteresisTest(unittest.TestCase):
    def test_hysteresis_holds_dense_mode_before_switching_off(self):
        gate = GateHysteresis(0.6, 0.4, minimum_hold_steps=2)
        self.assertEqual(gate.update(0.7), "dense")
        self.assertEqual(gate.update(0.2), "dense")
        self.assertEqual(gate.update(0.2), "standard")
        self.assertEqual(gate.switches, 2)

    def test_probability_between_thresholds_keeps_current_mode(self):
        gate = GateHysteresis(0.6, 0.4, minimum_hold_steps=0)
        self.assertEqual(gate.update(0.5), "standard")
        self.assertEqual(gate.update(0.7), "dense")
        self.assertEqual(gate.update(0.5), "dense")

    def test_invalid_threshold_order_is_rejected(self):
        with self.assertRaises(ValueError):
            GateHysteresis(0.4, 0.6, minimum_hold_steps=1)


class _FixedPolicy(object):
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)
        self.calls = 0

    def get_action(self, state):
        self.calls += 1
        return self.action.copy()


class TemporalGateControllerTest(unittest.TestCase):
    def test_feature_dimension_infers_actor_augmented_tracks(self):
        self.assertEqual(infer_max_tracks(76), 4)
        self.assertEqual(infer_max_tracks(82, actor_feature_dim=6), 4)
        with self.assertRaises(ValueError):
            infer_max_tracks(80, actor_feature_dim=6)

    def test_temporal_checkpoint_uses_history_and_evaluation_stride(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            detector = LocalRobotDetector()
            detector_path = directory / "detector.pt"
            torch.save(
                {
                    "model_config": {},
                    "model_state_dict": detector.state_dict(),
                },
                detector_path,
            )
            gate = TemporalInteractionGate(input_dim=82, hidden_dim=8)
            for parameter in gate.parameters():
                torch.nn.init.zeros_(parameter)
            gate.head[-1].bias.data.fill_(10.0)
            gate_path = directory / "gate.pt"
            torch.save(
                {
                    "model_id": "T1",
                    "feature_set": "base_and_actor_actions",
                    "model_config": {"input_dim": 82, "hidden_dim": 8},
                    "model_state_dict": gate.state_dict(),
                    "feature_mean": np.zeros(82, dtype=np.float32),
                    "feature_std": np.ones(82, dtype=np.float32),
                    "threshold": 0.5,
                    "sequence_length": 8,
                },
                gate_path,
            )
            standard = _FixedPolicy([-0.5, 0.1])
            dense = _FixedPolicy([0.5, -0.2])
            controller = LearnedInteractionGateController(
                standard,
                dense,
                detector_path,
                gate_path,
                "cpu",
                switch_off_threshold=0.4,
                minimum_hold_steps=0,
                evaluation_stride=2,
            )
            controller.reset(["r1"])
            position = SimpleNamespace(x=0.0, y=0.0)
            env = SimpleNamespace(
                last_odom={
                    "r1": SimpleNamespace(
                        pose=SimpleNamespace(
                            pose=SimpleNamespace(position=position)
                        )
                    )
                },
                raw_lidar_points={"r1": np.empty((0, 3), dtype=np.float32)},
                _get_robot_yaw=lambda name: 0.0,
            )
            state = np.zeros(24, dtype=np.float32)

            action, mode, probability, _ = controller.choose_action(
                env, "r1", state, 0.2
            )
            np.testing.assert_array_equal(action, dense.action)
            self.assertEqual(mode, "dense")
            self.assertGreater(probability, 0.99)
            self.assertEqual(len(controller.feature_histories["r1"]), 1)

            controller.choose_action(env, "r1", state, 0.4)
            self.assertEqual(len(controller.feature_histories["r1"]), 1)
            controller.choose_action(env, "r1", state, 0.6)
            self.assertEqual(len(controller.feature_histories["r1"]), 2)
            self.assertEqual(standard.calls, 2)
            self.assertEqual(dense.calls, 3)

    def test_base_only_single_frame_gate_uses_one_actor_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            detector = LocalRobotDetector()
            detector_path = directory / "detector.pt"
            torch.save(
                {"model_config": {}, "model_state_dict": detector.state_dict()},
                detector_path,
            )
            gate = TemporalInteractionGate(input_dim=76, hidden_dim=8)
            for parameter in gate.parameters():
                torch.nn.init.zeros_(parameter)
            gate.head[-1].bias.data.fill_(10.0)
            gate_path = directory / "gate.pt"
            torch.save(
                {
                    "model_id": "T1",
                    "feature_set": "base",
                    "model_config": {"input_dim": 76, "hidden_dim": 8},
                    "model_state_dict": gate.state_dict(),
                    "feature_mean": np.zeros(76, dtype=np.float32),
                    "feature_std": np.ones(76, dtype=np.float32),
                    "threshold": 0.5,
                    "sequence_length": 1,
                },
                gate_path,
            )
            standard = _FixedPolicy([-0.5, 0.1])
            dense = _FixedPolicy([0.5, -0.2])
            controller = LearnedInteractionGateController(
                standard,
                dense,
                detector_path,
                gate_path,
                "cpu",
                switch_off_threshold=0.4,
                minimum_hold_steps=0,
            )
            controller.reset(["r1"])
            position = SimpleNamespace(x=0.0, y=0.0)
            env = SimpleNamespace(
                last_odom={
                    "r1": SimpleNamespace(
                        pose=SimpleNamespace(
                            pose=SimpleNamespace(position=position)
                        )
                    )
                },
                raw_lidar_points={"r1": np.empty((0, 3), dtype=np.float32)},
                _get_robot_yaw=lambda name: 0.0,
            )
            action, mode, _, _ = controller.choose_action(
                env, "r1", np.zeros(24, dtype=np.float32), 0.2
            )
            np.testing.assert_array_equal(action, dense.action)
            self.assertEqual(mode, "dense")
            self.assertEqual(len(controller.feature_histories["r1"]), 1)
            self.assertEqual(standard.calls, 0)
            self.assertEqual(dense.calls, 1)


if __name__ == "__main__":
    unittest.main()
