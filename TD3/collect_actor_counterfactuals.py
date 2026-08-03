#!/usr/bin/env python3
import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import rospy
import torch

from actor_counterfactual import (
    LABEL_AMBIGUOUS,
    choose_actor_distribution_label,
    choose_actor_label,
    counterfactual_repeatability,
    distribution_label_repeatability,
)
from actor_models import Actor
from geometry_msgs.msg import Twist
from multi_agent_velodyne_env import MultiAgentGazeboEnv
from robot_perception.dataset import build_frame_examples
from robot_perception.gate_features import build_gate_feature, gate_feature_dim
from robot_perception.models import LocalRobotDetector
from robot_perception.tracker import RobotCandidateTracker


STATE_DIM = 24
ACTION_DIM = 2
ENVIRONMENT_DIM = 20
AGENT_NAMES = ("r1", "r2", "r3", "r4", "r5")
OUTCOME_KEYS = (
    "ego_collision",
    "collision_count",
    "ego_target",
    "minimum_ego_clearance",
    "ego_progress",
    "steps",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect same-state short-rollout labels for the frozen actor Gate."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generalist-actor", type=Path, required=True)
    parser.add_argument("--strong-actor", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--launchfile", default="multi_robot_scenario_strong_interaction_pilot_5.launch")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--anchor-start-step", type=int, default=0)
    parser.add_argument("--anchor-stride", type=int, default=4)
    parser.add_argument("--max-anchors-per-episode", type=int, default=4)
    parser.add_argument("--agents-per-anchor", type=int, default=2)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--max-tracks", type=int, default=4)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--repeat-baseline", action="store_true")
    parser.add_argument("--rollouts-per-actor", type=int, default=1)
    parser.add_argument("--label-batches", type=int, default=1)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


class FrozenActor:
    def __init__(self, checkpoint, device):
        self.device = torch.device(device)
        self.actor = Actor(STATE_DIM, ACTION_DIM).to(self.device)
        state = torch.load(checkpoint, map_location=self.device)
        self.actor.load_state_dict(state)
        self.actor.eval()

    @torch.no_grad()
    def action(self, state):
        value = torch.from_numpy(np.asarray(state, dtype=np.float32)).reshape(1, -1)
        return self.actor(value.to(self.device)).cpu().numpy().reshape(-1)


@torch.no_grad()
def detector_probabilities(detector, patches, device):
    if len(patches) == 0:
        return np.empty((0,), dtype=np.float32)
    values = torch.from_numpy(patches.astype(np.float32)).to(device)
    return torch.sigmoid(detector(values)[0]).cpu().numpy()


def agent_pose(env, name):
    odom = env.last_odom[name]
    return np.asarray(
        [
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            env._get_robot_yaw(name),
        ],
        dtype=np.float32,
    )


def gate_features_for_frame(env, states, active_mask, trackers, detector, args, logical_time):
    features = {}
    for index, name in enumerate(AGENT_NAMES):
        if not active_mask[index]:
            continue
        pose = agent_pose(env, name)
        examples = build_frame_examples(
            env.raw_lidar_points[name],
            pose,
            [],
            max_background_candidates=args.max_candidates,
        )
        probabilities = detector_probabilities(detector, examples.patches, args.device)
        tracked = trackers[name].update(
            examples.candidate_centers,
            probabilities,
            pose,
            logical_time,
        )
        features[index] = build_gate_feature(
            states[index], tracked, max_tracks=args.max_tracks
        )
    return features


def nearest_robot_distance(env, ego_index, active_mask):
    ego_name = AGENT_NAMES[ego_index]
    distances = [
        float(np.linalg.norm(env.robot_positions[name] - env.robot_positions[ego_name]))
        for index, name in enumerate(AGENT_NAMES)
        if index != ego_index and active_mask[index]
    ]
    return min(distances) if distances else 10.0


def select_anchor_agents(env, active_mask, limit, offset):
    ranked = sorted(
        (
            (nearest_robot_distance(env, index, active_mask), index)
            for index in range(len(AGENT_NAMES))
            if active_mask[index]
        ),
        key=lambda item: item[0],
    )
    if not ranked:
        return []
    selected = [ranked[0][1]]
    remaining = [index for _, index in ranked[1:]]
    while len(selected) < min(limit, len(ranked)) and remaining:
        selected.append(remaining[(offset + len(selected) - 1) % len(remaining)])
    return selected


def actor_actions(states, active_mask, generalist, strong=None, strong_ego=None):
    actions = []
    raw_actions = []
    for index, state in enumerate(states):
        if not active_mask[index]:
            raw = np.zeros(2, dtype=np.float32)
        else:
            policy = strong if index == strong_ego else generalist
            raw = policy.action(state)
        raw_actions.append(raw)
        actions.append([float((raw[0] + 1.0) / 2.0), float(raw[1])])
    return actions, raw_actions


def ego_clearance(env, ego_index, reference_active_mask):
    ego_name = AGENT_NAMES[ego_index]
    distances = [
        float(np.linalg.norm(env.robot_positions[name] - env.robot_positions[ego_name]))
        for index, name in enumerate(AGENT_NAMES)
        if index != ego_index and reference_active_mask[index]
    ]
    return min(distances) if distances else 10.0


def run_branch(
    env,
    states,
    active_mask,
    ego_index,
    generalist,
    strong,
    horizon,
    use_strong,
    initial_actor_states,
):
    reference_active_mask = list(active_mask)
    branch_active_mask = list(active_mask)
    initial_distance = float(states[ego_index][-4])
    minimum_clearance = ego_clearance(env, ego_index, reference_active_mask)
    collision_agents = set()
    ego_target = False
    final_distance = initial_distance
    initial_raw_action = None
    steps = 0
    for _ in range(horizon):
        policy_states = initial_actor_states if steps == 0 else states
        actions, raw_actions = actor_actions(
            policy_states,
            branch_active_mask,
            generalist,
            strong=strong,
            strong_ego=ego_index if use_strong else None,
        )
        if initial_raw_action is None:
            initial_raw_action = raw_actions[ego_index].copy()
        states, _, dones, targets, collisions = env.step(actions, branch_active_mask)
        steps += 1
        minimum_clearance = min(
            minimum_clearance,
            ego_clearance(env, ego_index, reference_active_mask),
        )
        final_distance = float(states[ego_index][-4])
        for index, collision in enumerate(collisions):
            if collision and branch_active_mask[index]:
                collision_agents.add(index)
        ego_target = bool(ego_target or targets[ego_index])
        for index, done in enumerate(dones):
            if done and branch_active_mask[index]:
                branch_active_mask[index] = False
        if not branch_active_mask[ego_index]:
            break
    outcome = {
        "ego_collision": bool(ego_index in collision_agents),
        "collision_count": int(len(collision_agents)),
        "ego_target": ego_target,
        "minimum_ego_clearance": float(minimum_clearance),
        "ego_progress": float(initial_distance - final_distance),
        "steps": int(steps),
    }
    return outcome, np.asarray(initial_raw_action, dtype=np.float32)


def outcome_vector(outcome):
    return np.asarray([float(outcome[key]) for key in OUTCOME_KEYS], dtype=np.float32)


def controller_motion(env):
    maximum_linear = 0.0
    maximum_angular = 0.0
    for odom in env.last_odom.values():
        if odom is None:
            return float("inf"), float("inf")
        twist = odom.twist.twist
        maximum_linear = max(
            maximum_linear,
            float(np.hypot(twist.linear.x, twist.linear.y)),
        )
        maximum_angular = max(maximum_angular, abs(float(twist.angular.z)))
    return maximum_linear, maximum_angular


def settle_controllers(env, timeout=4.0):
    zero = Twist()
    if env.fixed_physics_step_size is not None:
        settle_step = 0.2
        elapsed_simulation = 0.0
        stable_samples = 0
        while True:
            env._pause_fixed_physics()
            for _ in range(3):
                for publisher in env.vel_pubs.values():
                    publisher.publish(zero)
                time.sleep(0.02)
            previous_velodyne_counts = dict(env.velodyne_update_counts)
            previous_odom_counts = dict(env.odom_update_counts)
            target_time = env._advance_fixed_physics(settle_step)
            env._wait_for_fixed_sensor_updates(
                previous_velodyne_counts,
                previous_odom_counts,
                target_time,
            )
            env.wait_for_all_odom()
            maximum_linear, maximum_angular = controller_motion(env)
            if maximum_linear <= 0.01 and maximum_angular <= 0.02:
                stable_samples += 1
            else:
                stable_samples = 0
            elapsed_simulation += settle_step
            if stable_samples >= 2:
                return
            if elapsed_simulation >= timeout:
                raise RuntimeError(
                    "controller did not settle after %.1f simulation seconds: "
                    "linear=%.4f angular=%.4f"
                    % (elapsed_simulation, maximum_linear, maximum_angular)
                )

    rospy.wait_for_service("/gazebo/unpause_physics")
    env.unpause()
    started = time.monotonic()
    stable_samples = 0
    while True:
        for publisher in env.vel_pubs.values():
            publisher.publish(zero)
        time.sleep(0.05)
        maximum_linear, maximum_angular = controller_motion(env)
        if maximum_linear <= 0.01 and maximum_angular <= 0.02:
            stable_samples += 1
        else:
            stable_samples = 0
        elapsed = time.monotonic() - started
        if elapsed >= 0.2 and stable_samples >= 3:
            break
        if elapsed >= timeout:
            raise RuntimeError(
                "controller did not settle after %.1f wall-clock seconds: "
                "linear=%.4f angular=%.4f"
                % (elapsed, maximum_linear, maximum_angular)
            )
    rospy.wait_for_service("/gazebo/pause_physics")
    env.pause()


def reset_to_case(env, case_index):
    if all(odom is not None for odom in env.last_odom.values()):
        settle_controllers(env)
    env.curriculum_case_index = int(case_index)
    return env.reset(), [True] * len(AGENT_NAMES)


def replay_to_anchor(env, case_index, action_prefix):
    states, active_mask = reset_to_case(env, case_index)
    for recorded_actions in action_prefix:
        states, _, dones, _, _ = env.step(recorded_actions, active_mask)
        for index, done in enumerate(dones):
            if done and active_mask[index]:
                active_mask[index] = False
    return states, active_mask


def physics_snapshot(env, states):
    snapshot = {}
    for index, name in enumerate(AGENT_NAMES):
        odom = env.last_odom[name]
        snapshot[name] = {
            "position": np.asarray(env.robot_positions[name], dtype=np.float32).copy(),
            "yaw": float(env._get_robot_yaw(name)),
            "linear_velocity": np.asarray(
                [odom.twist.twist.linear.x, odom.twist.twist.linear.y],
                dtype=np.float32,
            ),
            "angular_velocity": float(odom.twist.twist.angular.z),
            "actor_state": np.asarray(states[index], dtype=np.float32).copy(),
        }
    return snapshot


def angle_error(first, second):
    return abs(float((first - second + np.pi) % (2.0 * np.pi) - np.pi))


def replay_alignment(
    env,
    reference_snapshot,
    reference_active_mask,
    states,
    active_mask,
):
    position_error = max(
        float(
            np.linalg.norm(
                env.robot_positions[name] - reference_snapshot[name]["position"]
            )
        )
        for name in AGENT_NAMES
    )
    yaw_error = max(
        angle_error(env._get_robot_yaw(name), reference_snapshot[name]["yaw"])
        for name in AGENT_NAMES
    )
    linear_velocity_error = max(
        float(
            np.linalg.norm(
                np.asarray(
                    [
                        env.last_odom[name].twist.twist.linear.x,
                        env.last_odom[name].twist.twist.linear.y,
                    ],
                    dtype=np.float32,
                )
                - reference_snapshot[name]["linear_velocity"]
            )
        )
        for name in AGENT_NAMES
    )
    angular_velocity_error = max(
        abs(
            float(env.last_odom[name].twist.twist.angular.z)
            - reference_snapshot[name]["angular_velocity"]
        )
        for name in AGENT_NAMES
    )
    actor_state_error = max(
        float(
            np.max(
                np.abs(
                    np.asarray(states[index], dtype=np.float32)
                    - reference_snapshot[name]["actor_state"]
                )
            )
        )
        for index, name in enumerate(AGENT_NAMES)
    )
    return {
        "active_mask_match": list(active_mask) == list(reference_active_mask),
        "position_error": position_error,
        "yaw_error": yaw_error,
        "linear_velocity_error": linear_velocity_error,
        "angular_velocity_error": angular_velocity_error,
        "actor_state_error": actor_state_error,
    }


def collect_branch_rollout(
    env,
    case_index,
    action_prefix,
    anchor_snapshot,
    anchor_active_mask,
    anchor_states,
    ego_index,
    generalist,
    strong,
    horizon,
    use_strong,
):
    branch_states, branch_active_mask = replay_to_anchor(
        env, case_index, action_prefix
    )
    alignment = replay_alignment(
        env,
        anchor_snapshot,
        anchor_active_mask,
        branch_states,
        branch_active_mask,
    )
    outcome, action = run_branch(
        env,
        branch_states,
        branch_active_mask,
        ego_index,
        generalist,
        strong,
        horizon,
        use_strong=use_strong,
        initial_actor_states=anchor_states,
    )
    return outcome, action, alignment


def scenario_metadata(env):
    case = env.current_curriculum_case or {}
    view = case.get("view", {})
    return (
        str(case.get("scenario_id") or case.get("name") or "unknown"),
        str(view.get("perception_pool") or case.get("preset") or "unknown"),
        str(view.get("interaction_band") or "unknown"),
    )


def write_shard(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def main():
    args = parse_args()
    integer_values = (
        args.episodes,
        args.horizon,
        args.anchor_stride,
        args.max_anchors_per_episode,
        args.agents_per_anchor,
        args.max_tracks,
        args.max_candidates,
        args.rollouts_per_actor,
        args.label_batches,
        args.bootstrap_resamples,
    )
    if min(integer_values) < 1:
        raise ValueError("counterfactual collection dimensions must be positive")
    if args.anchor_start_step < 0:
        raise ValueError("anchor-start-step must be non-negative")
    distribution_mode = args.rollouts_per_actor > 1 or args.label_batches > 1
    if distribution_mode and min(args.rollouts_per_actor, args.label_batches) < 2:
        raise ValueError(
            "distribution mode requires at least two rollouts and two label batches"
        )
    if distribution_mode and args.repeat_baseline:
        raise ValueError("repeat-baseline belongs to the single-rollout v1 audit")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ["DRL_MULTI_SCENARIO"] = "manifest"
    os.environ["DRL_MULTI_MANIFEST_PATH"] = str(args.manifest.resolve())
    os.environ["DRL_MULTI_MANIFEST_SAMPLING"] = "cycle"
    os.environ["DRL_MULTI_RECORD_RAW_LIDAR"] = "1"

    generalist = FrozenActor(args.generalist_actor, args.device)
    strong = FrozenActor(args.strong_actor, args.device)
    detector_checkpoint = torch.load(
        args.detector_checkpoint, map_location=args.device
    )
    detector = LocalRobotDetector(
        **detector_checkpoint.get("model_config", {})
    ).to(args.device)
    detector.load_state_dict(detector_checkpoint["model_state_dict"])
    detector.eval()

    env = MultiAgentGazeboEnv(
        args.launchfile,
        ENVIRONMENT_DIM,
        agent_names=list(AGENT_NAMES),
        cooperative_reward=False,
        robot_safe_distance=0.0,
        weak_coupling_layout=True,
        scenario_mode="manifest",
        fixed_physics_step_size=0.001,
    )
    totals = Counter()
    try:
        for episode in range(args.episodes):
            case_index = episode % len(env.curriculum_cases)
            states, active_mask = reset_to_case(env, case_index)
            scenario_id, scenario_pool, interaction_band = scenario_metadata(env)
            output_path = args.output_dir / (scenario_id.replace("/", "_") + ".npz")
            trackers = {name: RobotCandidateTracker() for name in AGENT_NAMES}
            records = []
            anchor_count = 0
            action_prefix = []
            for step in range(args.max_episode_steps):
                if not all(active_mask):
                    break
                features = gate_features_for_frame(
                    env,
                    states,
                    active_mask,
                    trackers,
                    detector,
                    args,
                    logical_time=step * 0.2,
                )
                anchor_due = bool(
                    step >= args.anchor_start_step
                    and (step - args.anchor_start_step) % args.anchor_stride == 0
                )
                if anchor_due and anchor_count < args.max_anchors_per_episode:
                    egos = select_anchor_agents(
                        env, active_mask, args.agents_per_anchor, anchor_count
                    )
                    if egos:
                        anchor_states = [
                            np.asarray(state, dtype=np.float32).copy() for state in states
                        ]
                        anchor_active_mask = list(active_mask)
                        anchor_snapshot = physics_snapshot(env, anchor_states)
                        anchor_nearest_distances = {
                            ego_index: nearest_robot_distance(
                                env, ego_index, anchor_active_mask
                            )
                            for ego_index in egos
                        }
                        for ego_index in egos:
                            alignments = []
                            distribution_record = None
                            if distribution_mode:
                                generalist_batches = []
                                strong_batches = []
                                batch_labels = []
                                batch_reasons = []
                                batch_diagnostics = []
                                generalist_action = None
                                strong_action = None
                                for batch_index in range(args.label_batches):
                                    generalist_outcomes = []
                                    strong_outcomes = []
                                    for _ in range(args.rollouts_per_actor):
                                        outcome, action, alignment = collect_branch_rollout(
                                            env,
                                            case_index,
                                            action_prefix,
                                            anchor_snapshot,
                                            anchor_active_mask,
                                            anchor_states,
                                            ego_index,
                                            generalist,
                                            strong,
                                            args.horizon,
                                            use_strong=False,
                                        )
                                        generalist_outcomes.append(outcome)
                                        alignments.append(alignment)
                                        if generalist_action is None:
                                            generalist_action = action
                                    for _ in range(args.rollouts_per_actor):
                                        outcome, action, alignment = collect_branch_rollout(
                                            env,
                                            case_index,
                                            action_prefix,
                                            anchor_snapshot,
                                            anchor_active_mask,
                                            anchor_states,
                                            ego_index,
                                            generalist,
                                            strong,
                                            args.horizon,
                                            use_strong=True,
                                        )
                                        strong_outcomes.append(outcome)
                                        alignments.append(alignment)
                                        if strong_action is None:
                                            strong_action = action
                                    batch_seed = (
                                        args.seed
                                        + episode * 100000
                                        + anchor_count * 1000
                                        + ego_index * 100
                                        + batch_index
                                    )
                                    batch_label, batch_reason, diagnostics = (
                                        choose_actor_distribution_label(
                                            generalist_outcomes,
                                            strong_outcomes,
                                            resamples=args.bootstrap_resamples,
                                            seed=batch_seed,
                                        )
                                    )
                                    generalist_batches.append(generalist_outcomes)
                                    strong_batches.append(strong_outcomes)
                                    batch_labels.append(batch_label)
                                    batch_reasons.append(batch_reason)
                                    batch_diagnostics.append(diagnostics)
                                label_stability = distribution_label_repeatability(
                                    batch_labels
                                )
                                flattened_generalist = [
                                    outcome
                                    for batch in generalist_batches
                                    for outcome in batch
                                ]
                                flattened_strong = [
                                    outcome
                                    for batch in strong_batches
                                    for outcome in batch
                                ]
                                generalist_outcome = {
                                    key: float(
                                        np.mean(
                                            [outcome[key] for outcome in flattened_generalist]
                                        )
                                    )
                                    for key in OUTCOME_KEYS
                                }
                                strong_outcome = {
                                    key: float(
                                        np.mean(
                                            [outcome[key] for outcome in flattened_strong]
                                        )
                                    )
                                    for key in OUTCOME_KEYS
                                }
                                repeatability = {
                                    "repeatable": label_stability["repeatable"],
                                    "discrete_match": len(set(batch_labels)) == 1,
                                    "clearance_delta": 0.0,
                                    "progress_delta": 0.0,
                                }
                                distribution_record = {
                                    "generalist_outcomes": np.asarray(
                                        [
                                            [outcome_vector(outcome) for outcome in batch]
                                            for batch in generalist_batches
                                        ],
                                        dtype=np.float32,
                                    ),
                                    "strong_outcomes": np.asarray(
                                        [
                                            [outcome_vector(outcome) for outcome in batch]
                                            for batch in strong_batches
                                        ],
                                        dtype=np.float32,
                                    ),
                                    "batch_labels": np.asarray(
                                        batch_labels, dtype=np.int8
                                    ),
                                    "batch_reasons": np.asarray(batch_reasons),
                                    "batch_diagnostics": np.asarray(
                                        [
                                            json.dumps(item, sort_keys=True)
                                            for item in batch_diagnostics
                                        ]
                                    ),
                                }
                            else:
                                (
                                    generalist_outcome,
                                    generalist_action,
                                    alignment,
                                ) = collect_branch_rollout(
                                    env,
                                    case_index,
                                    action_prefix,
                                    anchor_snapshot,
                                    anchor_active_mask,
                                    anchor_states,
                                    ego_index,
                                    generalist,
                                    strong,
                                    args.horizon,
                                    use_strong=False,
                                )
                                alignments.append(alignment)
                                repeatability = {
                                    "repeatable": True,
                                    "discrete_match": True,
                                    "clearance_delta": 0.0,
                                    "progress_delta": 0.0,
                                }
                                if args.repeat_baseline:
                                    (
                                        repeated_outcome,
                                        _,
                                        alignment,
                                    ) = collect_branch_rollout(
                                        env,
                                        case_index,
                                        action_prefix,
                                        anchor_snapshot,
                                        anchor_active_mask,
                                        anchor_states,
                                        ego_index,
                                        generalist,
                                        strong,
                                        args.horizon,
                                        use_strong=False,
                                    )
                                    alignments.append(alignment)
                                    repeatability = counterfactual_repeatability(
                                        generalist_outcome, repeated_outcome
                                    )
                                strong_outcome, strong_action, alignment = (
                                    collect_branch_rollout(
                                        env,
                                        case_index,
                                        action_prefix,
                                        anchor_snapshot,
                                        anchor_active_mask,
                                        anchor_states,
                                        ego_index,
                                        generalist,
                                        strong,
                                        args.horizon,
                                        use_strong=True,
                                    )
                                )
                                alignments.append(alignment)
                            maximum_position_error = max(
                                item["position_error"] for item in alignments
                            )
                            maximum_yaw_error = max(
                                item["yaw_error"] for item in alignments
                            )
                            maximum_linear_velocity_error = max(
                                item["linear_velocity_error"] for item in alignments
                            )
                            maximum_angular_velocity_error = max(
                                item["angular_velocity_error"] for item in alignments
                            )
                            maximum_actor_state_error = max(
                                item["actor_state_error"] for item in alignments
                            )
                            active_mask_match = all(
                                item["active_mask_match"] for item in alignments
                            )
                            anchor_is_repeatable = bool(
                                active_mask_match
                                and maximum_position_error <= 0.02
                                and maximum_yaw_error <= 0.02
                                and maximum_linear_velocity_error <= 0.02
                                and maximum_angular_velocity_error <= 0.03
                            )
                            repeatability["repeatable"] = bool(
                                repeatability["repeatable"]
                                and anchor_is_repeatable
                            )
                            if not anchor_is_repeatable:
                                label, reason = LABEL_AMBIGUOUS, "nonrepeatable_anchor"
                            elif distribution_mode and repeatability["repeatable"]:
                                label = label_stability["label"]
                                reason = (
                                    batch_reasons[0]
                                    if len(set(batch_reasons)) == 1
                                    else "stable_distribution"
                                )
                            elif distribution_mode:
                                label = LABEL_AMBIGUOUS
                                reason = (
                                    "ambiguous_distribution"
                                    if set(batch_labels) == {LABEL_AMBIGUOUS}
                                    else "unstable_distribution"
                                )
                            elif repeatability["repeatable"]:
                                label, reason = choose_actor_label(
                                    generalist_outcome, strong_outcome
                                )
                            else:
                                label, reason = LABEL_AMBIGUOUS, "nonrepeatable"
                            records.append(
                                {
                                    "feature": features[ego_index],
                                    "actor_state": anchor_states[ego_index],
                                    "ego_index": ego_index,
                                    "anchor_step": step,
                                    "nearest_robot_distance": anchor_nearest_distances[
                                        ego_index
                                    ],
                                    "label": label,
                                    "reason": reason,
                                    "generalist_outcome": outcome_vector(
                                        generalist_outcome
                                    ),
                                    "strong_outcome": outcome_vector(strong_outcome),
                                    "generalist_action": generalist_action,
                                    "strong_action": strong_action,
                                    "distribution": distribution_record,
                                    "repeatability": np.asarray(
                                        [
                                            float(repeatability["repeatable"]),
                                            float(repeatability["discrete_match"]),
                                            repeatability["clearance_delta"],
                                            repeatability["progress_delta"],
                                            maximum_position_error,
                                            maximum_yaw_error,
                                            maximum_linear_velocity_error,
                                            maximum_angular_velocity_error,
                                            maximum_actor_state_error,
                                            float(active_mask_match),
                                        ],
                                        dtype=np.float32,
                                    ),
                                }
                            )
                            totals[reason] += 1
                        states, active_mask = replay_to_anchor(
                            env, case_index, action_prefix
                        )
                        anchor_count += 1
                        if anchor_count >= args.max_anchors_per_episode:
                            break

                actions, _ = actor_actions(states, active_mask, generalist)
                action_prefix.append(
                    [[float(value) for value in action] for action in actions]
                )
                states, _, dones, _, _ = env.step(actions, active_mask)
                for index, done in enumerate(dones):
                    if done and active_mask[index]:
                        active_mask[index] = False
                if not any(active_mask):
                    break

            if records:
                payload = {
                    "format_version": np.asarray(
                        3 if distribution_mode else 2, dtype=np.int32
                    ),
                    "scenario_id": np.asarray(scenario_id),
                    "scenario_pool": np.asarray(scenario_pool),
                    "interaction_band": np.asarray(interaction_band),
                    "split": np.asarray(args.split),
                    "feature_dim": np.asarray(
                        gate_feature_dim(STATE_DIM, args.max_tracks), dtype=np.int32
                    ),
                    "outcome_keys": np.asarray(OUTCOME_KEYS),
                    "features": np.stack([item["feature"] for item in records]),
                    "actor_states": np.stack(
                        [item["actor_state"] for item in records]
                    ),
                    "ego_indices": np.asarray(
                        [item["ego_index"] for item in records], dtype=np.uint8
                    ),
                    "anchor_steps": np.asarray(
                        [item["anchor_step"] for item in records], dtype=np.int32
                    ),
                    "nearest_robot_distances": np.asarray(
                        [item["nearest_robot_distance"] for item in records],
                        dtype=np.float32,
                    ),
                    "labels": np.asarray(
                        [item["label"] for item in records], dtype=np.int8
                    ),
                    "label_reasons": np.asarray(
                        [item["reason"] for item in records]
                    ),
                    "generalist_outcomes": np.stack(
                        [item["generalist_outcome"] for item in records]
                    ),
                    "strong_outcomes": np.stack(
                        [item["strong_outcome"] for item in records]
                    ),
                    "generalist_actions": np.stack(
                        [item["generalist_action"] for item in records]
                    ),
                    "strong_actions": np.stack(
                        [item["strong_action"] for item in records]
                    ),
                    "repeatability": np.stack(
                        [item["repeatability"] for item in records]
                    ),
                    "repeatability_keys": np.asarray(
                        (
                            "repeatable",
                            "discrete_match",
                            "clearance_delta",
                            "progress_delta",
                            "anchor_position_error",
                            "anchor_yaw_error",
                            "anchor_linear_velocity_error",
                            "anchor_angular_velocity_error",
                            "anchor_actor_state_error",
                            "anchor_active_mask_match",
                        )
                    ),
                    "horizon": np.asarray(args.horizon, dtype=np.int32),
                    "seed": np.asarray(args.seed, dtype=np.int64),
                }
                if distribution_mode:
                    payload.update(
                        {
                            "distribution_generalist_outcomes": np.stack(
                                [
                                    item["distribution"]["generalist_outcomes"]
                                    for item in records
                                ]
                            ),
                            "distribution_strong_outcomes": np.stack(
                                [
                                    item["distribution"]["strong_outcomes"]
                                    for item in records
                                ]
                            ),
                            "distribution_batch_labels": np.stack(
                                [
                                    item["distribution"]["batch_labels"]
                                    for item in records
                                ]
                            ),
                            "distribution_batch_reasons": np.stack(
                                [
                                    item["distribution"]["batch_reasons"]
                                    for item in records
                                ]
                            ),
                            "distribution_batch_diagnostics": np.stack(
                                [
                                    item["distribution"]["batch_diagnostics"]
                                    for item in records
                                ]
                            ),
                            "rollouts_per_actor": np.asarray(
                                args.rollouts_per_actor, dtype=np.int32
                            ),
                            "label_batches": np.asarray(
                                args.label_batches, dtype=np.int32
                            ),
                            "bootstrap_resamples": np.asarray(
                                args.bootstrap_resamples, dtype=np.int32
                            ),
                        }
                    )
                write_shard(output_path, payload)
            print(
                "episode=%d scenario=%s anchors=%d samples=%d labels=%s"
                % (
                    episode + 1,
                    scenario_id,
                    anchor_count,
                    len(records),
                    dict(Counter(item["reason"] for item in records)),
                ),
                flush=True,
            )
    finally:
        env.close()
    print(json.dumps({"samples": int(sum(totals.values())), "reasons": totals}, indent=2))


if __name__ == "__main__":
    main()
