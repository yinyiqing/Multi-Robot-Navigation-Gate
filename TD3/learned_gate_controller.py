from collections import deque

import numpy as np
import torch

from interaction_gate import InteractionGate
from robot_perception.dataset import build_frame_examples
from robot_perception.gate_features import (
    GLOBAL_FEATURE_DIM,
    TRACK_FEATURE_DIM,
    build_gate_feature,
)
from robot_perception.models import LocalRobotDetector
from robot_perception.tracker import RobotCandidateTracker
from temporal_interaction_gate import (
    ACTOR_COMPARISON_DIM,
    TemporalInteractionGate,
    actor_comparison_features,
)


def infer_max_tracks(feature_dim, actor_state_dim=24, actor_feature_dim=0):
    track_features = (
        int(feature_dim)
        - int(actor_state_dim)
        - int(actor_feature_dim)
        - GLOBAL_FEATURE_DIM
    )
    if track_features < TRACK_FEATURE_DIM or track_features % TRACK_FEATURE_DIM:
        raise ValueError("Gate feature dimension cannot determine max_tracks")
    return track_features // TRACK_FEATURE_DIM


class GateHysteresis(object):
    def __init__(self, switch_on_threshold, switch_off_threshold, minimum_hold_steps):
        self.switch_on_threshold = float(switch_on_threshold)
        self.switch_off_threshold = float(switch_off_threshold)
        self.minimum_hold_steps = int(minimum_hold_steps)
        if not 0.0 <= self.switch_off_threshold <= self.switch_on_threshold <= 1.0:
            raise ValueError(
                "gate thresholds must satisfy 0 <= switch_off <= switch_on <= 1"
            )
        if self.minimum_hold_steps < 0:
            raise ValueError("minimum_hold_steps must be non-negative")
        self.reset()

    def reset(self):
        self.mode = "standard"
        self.dense_steps = 0
        self.switches = 0

    def update(self, probability):
        probability = float(probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("gate probability must lie within [0, 1]")
        if self.mode == "standard":
            if probability >= self.switch_on_threshold:
                self.mode = "dense"
                self.dense_steps = 0
                self.switches += 1
        else:
            self.dense_steps += 1
            if (
                self.dense_steps >= self.minimum_hold_steps
                and probability <= self.switch_off_threshold
            ):
                self.mode = "standard"
                self.dense_steps = 0
                self.switches += 1
        return self.mode


class LearnedInteractionGateController(object):
    """Run the frozen perception stack and Gate using deployable ego observations."""

    def __init__(
        self,
        standard_policy,
        dense_policy,
        detector_checkpoint,
        gate_checkpoint,
        device,
        switch_on_threshold=None,
        switch_off_threshold=None,
        minimum_hold_steps=3,
        max_candidates=12,
        evaluation_stride=1,
    ):
        self.standard_policy = standard_policy
        self.dense_policy = dense_policy
        self.device = torch.device(device)
        self.max_candidates = int(max_candidates)
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be positive")

        detector_payload = torch.load(
            detector_checkpoint, map_location=self.device, weights_only=False
        )
        self.detector = LocalRobotDetector(
            **detector_payload.get("model_config", {})
        ).to(self.device)
        self.detector.load_state_dict(detector_payload["model_state_dict"])
        self.detector.eval()

        gate_payload = torch.load(
            gate_checkpoint, map_location=self.device, weights_only=False
        )
        self.model_id = str(gate_payload.get("model_id", "legacy"))
        self.feature_set = str(gate_payload.get("feature_set", "base"))
        self.uses_actor_features = self.feature_set == "base_and_actor_actions"
        self.is_temporal = self.model_id == "T1"
        if self.is_temporal:
            self.gate = TemporalInteractionGate(
                **gate_payload["model_config"]
            ).to(self.device)
        else:
            self.gate = InteractionGate(**gate_payload["model_config"]).to(self.device)
        self.gate.load_state_dict(gate_payload["model_state_dict"])
        self.gate.eval()
        self.feature_mean = np.asarray(
            gate_payload["feature_mean"], dtype=np.float32
        )
        self.feature_std = np.asarray(gate_payload["feature_std"], dtype=np.float32)
        input_dim = int(gate_payload["model_config"]["input_dim"])
        if self.feature_mean.shape != (input_dim,) or self.feature_std.shape != (
            input_dim,
        ):
            raise ValueError("Gate normalization does not match its input dimension")
        if (
            not np.all(np.isfinite(self.feature_mean))
            or not np.all(np.isfinite(self.feature_std))
            or np.any(self.feature_std <= 0.0)
        ):
            raise ValueError("Gate normalization must be finite with positive scales")
        if self.is_temporal and not self.uses_actor_features:
            raise ValueError("T1 Gate requires Actor comparison features")
        actor_feature_dim = ACTOR_COMPARISON_DIM if self.uses_actor_features else 0
        inferred_max_tracks = infer_max_tracks(
            input_dim, actor_feature_dim=actor_feature_dim
        )
        self.max_tracks = int(gate_payload.get("max_tracks", inferred_max_tracks))
        if self.max_tracks != inferred_max_tracks:
            raise ValueError("Gate max_tracks does not match its feature dimension")
        self.sequence_length = int(gate_payload.get("sequence_length", 1))
        if self.is_temporal and self.sequence_length < 2:
            raise ValueError("temporal Gate requires sequence_length >= 2")
        if not self.is_temporal and self.sequence_length != 1:
            raise ValueError("static Gate requires sequence_length == 1")
        self.evaluation_stride = int(evaluation_stride)
        if self.evaluation_stride < 1:
            raise ValueError("evaluation_stride must be positive")
        checkpoint_threshold = float(gate_payload["threshold"])
        self.switch_on_threshold = (
            checkpoint_threshold
            if switch_on_threshold is None
            else float(switch_on_threshold)
        )
        self.switch_off_threshold = (
            max(self.switch_on_threshold - 0.10, 0.0)
            if switch_off_threshold is None
            else float(switch_off_threshold)
        )
        self.minimum_hold_steps = int(minimum_hold_steps)
        self.trackers = {}
        self.switchers = {}
        self.feature_histories = {}
        self.evaluation_steps = {}
        self.last_probabilities = {}
        self.last_track_counts = {}
        self.probability_sum = 0.0
        self.probability_count = 0

    def reset(self, agent_names):
        self.trackers = {name: RobotCandidateTracker() for name in agent_names}
        self.switchers = {
            name: GateHysteresis(
                self.switch_on_threshold,
                self.switch_off_threshold,
                self.minimum_hold_steps,
            )
            for name in agent_names
        }
        self.feature_histories = {
            name: deque(maxlen=self.sequence_length) for name in agent_names
        }
        self.evaluation_steps = {name: 0 for name in agent_names}
        self.last_probabilities = {name: 0.0 for name in agent_names}
        self.last_track_counts = {name: 0 for name in agent_names}
        self.probability_sum = 0.0
        self.probability_count = 0

    @torch.no_grad()
    def _detector_probabilities(self, patches):
        if len(patches) == 0:
            return np.empty((0,), dtype=np.float32)
        values = torch.from_numpy(patches.astype(np.float32)).to(self.device)
        return torch.sigmoid(self.detector(values)[0]).cpu().numpy()

    @torch.no_grad()
    def _gate_probability(self, name, feature):
        normalized = (feature - self.feature_mean) / self.feature_std
        if self.is_temporal:
            history = self.feature_histories[name]
            history.append(normalized.astype(np.float32))
            window = np.zeros(
                (self.sequence_length, len(normalized)), dtype=np.float32
            )
            window[-len(history) :] = np.asarray(history, dtype=np.float32)
            values = torch.from_numpy(window[None, ...]).to(self.device)
        else:
            values = torch.from_numpy(normalized.reshape(1, -1)).to(self.device)
        return float(torch.sigmoid(self.gate(values)).cpu().item())

    def choose_action(self, env, name, state, logical_time):
        if name not in self.trackers or name not in self.switchers:
            raise ValueError("learned Gate must be reset before use")
        evaluate_gate = self.evaluation_steps[name] % self.evaluation_stride == 0
        self.evaluation_steps[name] += 1
        if evaluate_gate:
            odom = env.last_odom[name]
            pose = np.asarray(
                [
                    odom.pose.pose.position.x,
                    odom.pose.pose.position.y,
                    env._get_robot_yaw(name),
                ],
                dtype=np.float32,
            )
            examples = build_frame_examples(
                env.raw_lidar_points[name],
                pose,
                [],
                max_background_candidates=self.max_candidates,
            )
            probabilities = self._detector_probabilities(examples.patches)
            tracked = self.trackers[name].update(
                examples.candidate_centers,
                probabilities,
                pose,
                float(logical_time),
            )
            state_array = np.asarray(state)
            standard_action = self.standard_policy.get_action(state_array)
            dense_action = self.dense_policy.get_action(state_array)
            feature = build_gate_feature(
                state_array, tracked, max_tracks=self.max_tracks
            )
            if self.uses_actor_features:
                action_features = actor_comparison_features(
                    np.asarray(standard_action, dtype=np.float32)[None, :],
                    np.asarray(dense_action, dtype=np.float32)[None, :],
                )[0]
                feature = np.concatenate((feature, action_features)).astype(
                    np.float32
                )
            probability = self._gate_probability(name, feature)
            mode = self.switchers[name].update(probability)
            self.last_probabilities[name] = probability
            self.last_track_counts[name] = len(tracked)
            action = dense_action if mode == "dense" else standard_action
        else:
            probability = self.last_probabilities[name]
            mode = self.switchers[name].mode
            policy = self.dense_policy if mode == "dense" else self.standard_policy
            action = policy.get_action(np.asarray(state))
        self.probability_sum += probability
        self.probability_count += 1
        return action, mode, probability, self.last_track_counts[name]

    def episode_stats(self):
        return {
            "switches": int(sum(item.switches for item in self.switchers.values())),
            "mean_probability": (
                self.probability_sum / self.probability_count
                if self.probability_count
                else 0.0
            ),
        }
