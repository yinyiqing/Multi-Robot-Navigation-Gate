import numpy as np
import torch

from interaction_gate import InteractionGate
from robot_perception.dataset import build_frame_examples
from robot_perception.gate_features import build_gate_feature
from robot_perception.models import LocalRobotDetector
from robot_perception.tracker import RobotCandidateTracker


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
        self.gate = InteractionGate(**gate_payload["model_config"]).to(self.device)
        self.gate.load_state_dict(gate_payload["model_state_dict"])
        self.gate.eval()
        self.feature_mean = np.asarray(
            gate_payload["feature_mean"], dtype=np.float32
        )
        self.feature_std = np.asarray(gate_payload["feature_std"], dtype=np.float32)
        self.max_tracks = int(gate_payload["max_tracks"])
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
        self.probability_sum = 0.0
        self.probability_count = 0

    @torch.no_grad()
    def _detector_probabilities(self, patches):
        if len(patches) == 0:
            return np.empty((0,), dtype=np.float32)
        values = torch.from_numpy(patches.astype(np.float32)).to(self.device)
        return torch.sigmoid(self.detector(values)[0]).cpu().numpy()

    @torch.no_grad()
    def _gate_probability(self, feature):
        normalized = (feature - self.feature_mean) / self.feature_std
        values = torch.from_numpy(normalized.reshape(1, -1)).to(self.device)
        return float(torch.sigmoid(self.gate(values)).cpu().item())

    def choose_action(self, env, name, state, logical_time):
        if name not in self.trackers or name not in self.switchers:
            raise ValueError("learned Gate must be reset before use")
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
        feature = build_gate_feature(state, tracked, max_tracks=self.max_tracks)
        probability = self._gate_probability(feature)
        mode = self.switchers[name].update(probability)
        self.probability_sum += probability
        self.probability_count += 1
        policy = self.dense_policy if mode == "dense" else self.standard_policy
        action = policy.get_action(np.asarray(state))
        return action, mode, probability, len(tracked)

    def episode_stats(self):
        return {
            "switches": int(sum(item.switches for item in self.switchers.values())),
            "mean_probability": (
                self.probability_sum / self.probability_count
                if self.probability_count
                else 0.0
            ),
        }
