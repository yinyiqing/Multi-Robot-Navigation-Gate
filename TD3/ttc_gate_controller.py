from collections import namedtuple

import numpy as np
import torch

from robot_perception.dataset import build_frame_examples
from robot_perception.models import LocalRobotDetector
from robot_perception.tracker import RobotCandidateTracker


RiskThresholds = namedtuple(
    "RiskThresholds",
    (
        "shape_probability",
        "minimum_age",
        "maximum_range",
        "minimum_closing_speed",
        "maximum_ttc",
        "maximum_cpa_distance",
    ),
)


def candidate_matches_risk(candidate, thresholds):
    shape_probability = max(
        float(candidate.shape_probability),
        float(candidate.smoothed_shape_probability),
    )
    return bool(
        shape_probability >= thresholds.shape_probability
        and int(candidate.age) >= thresholds.minimum_age
        and float(np.linalg.norm(candidate.local_center)) <= thresholds.maximum_range
        and float(candidate.closing_speed) >= thresholds.minimum_closing_speed
        and float(candidate.ttc) <= thresholds.maximum_ttc
        and float(candidate.closest_approach_distance)
        <= thresholds.maximum_cpa_distance
    )


class TtcCpaActorSwitcher(object):
    """Deployable TTC/CPA rule using the frozen detector and candidate tracker."""

    def __init__(
        self,
        standard_policy,
        interaction_policy,
        detector_checkpoint,
        device,
        enter_thresholds,
        stay_thresholds,
        minimum_hold_steps=3,
        max_candidates=12,
        evaluation_stride=2,
    ):
        self.standard_policy = standard_policy
        self.interaction_policy = interaction_policy
        self.device = torch.device(device)
        self.enter_thresholds = enter_thresholds
        self.stay_thresholds = stay_thresholds
        self.minimum_hold_steps = int(minimum_hold_steps)
        self.max_candidates = int(max_candidates)
        self.evaluation_stride = int(evaluation_stride)
        if self.minimum_hold_steps < 0:
            raise ValueError("minimum_hold_steps cannot be negative")
        if self.max_candidates < 1 or self.evaluation_stride < 1:
            raise ValueError("candidate count and evaluation stride must be positive")

        payload = torch.load(
            detector_checkpoint, map_location=self.device, weights_only=False
        )
        self.detector = LocalRobotDetector(**payload.get("model_config", {})).to(
            self.device
        )
        self.detector.load_state_dict(payload["model_state_dict"])
        self.detector.eval()
        self.trackers = {}
        self.modes = {}
        self.mode_steps = {}
        self.evaluation_steps = {}
        self.last_risks = {}
        self.last_track_counts = {}
        self.switch_count = 0
        self.risk_sum = 0.0
        self.risk_count = 0

    def reset(self, agent_names):
        self.trackers = {name: RobotCandidateTracker() for name in agent_names}
        self.modes = {name: "standard" for name in agent_names}
        self.mode_steps = {name: 0 for name in agent_names}
        self.evaluation_steps = {name: 0 for name in agent_names}
        self.last_risks = {name: False for name in agent_names}
        self.last_track_counts = {name: 0 for name in agent_names}
        self.switch_count = 0
        self.risk_sum = 0.0
        self.risk_count = 0

    @torch.no_grad()
    def _detector_probabilities(self, patches):
        if len(patches) == 0:
            return np.empty((0,), dtype=np.float32)
        values = torch.from_numpy(patches.astype(np.float32)).to(self.device)
        return torch.sigmoid(self.detector(values)[0]).cpu().numpy()

    def _update_mode(self, name, tracked):
        mode = self.modes[name]
        thresholds = (
            self.enter_thresholds if mode == "standard" else self.stay_thresholds
        )
        risk = any(candidate_matches_risk(item, thresholds) for item in tracked)
        next_mode = mode
        if mode == "standard" and risk:
            next_mode = "dense"
        elif (
            mode == "dense"
            and self.mode_steps[name] >= self.minimum_hold_steps
            and not risk
        ):
            next_mode = "standard"
        if next_mode != mode:
            self.switch_count += 1
            self.mode_steps[name] = 0
        self.modes[name] = next_mode
        self.mode_steps[name] += 1
        self.last_risks[name] = risk
        return next_mode, risk

    def choose_action(self, env, name, state, logical_time):
        if name not in self.trackers:
            raise ValueError("TTC/CPA rule must be reset before use")
        evaluate_rule = self.evaluation_steps[name] % self.evaluation_stride == 0
        self.evaluation_steps[name] += 1
        if evaluate_rule:
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
            mode, risk = self._update_mode(name, tracked)
            self.last_track_counts[name] = len(tracked)
        else:
            mode = self.modes[name]
            risk = self.last_risks[name]
        policy = self.interaction_policy if mode == "dense" else self.standard_policy
        action = policy.get_action(np.asarray(state))
        self.risk_sum += float(risk)
        self.risk_count += 1
        return action, mode, float(risk), self.last_track_counts[name]

    def episode_stats(self):
        return {
            "switches": int(self.switch_count),
            "mean_probability": (
                self.risk_sum / self.risk_count if self.risk_count else 0.0
            ),
        }
