#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

from actor_models import Actor
from g27_counterfactual import (
    AGENT_NAMES,
    alignment_metrics,
    base_reward_components,
    canonical_sha256,
    deployed_action,
    validate_branch_spec,
)
from multi_agent_velodyne_env import MultiAgentGazeboEnv
from robot_perception.dataset import build_frame_examples
from robot_perception.gate_features import build_gate_feature
from robot_perception.models import LocalRobotDetector
from robot_perception.tracker import RobotCandidateTracker
from temporal_interaction_gate import actor_comparison_features


STATE_DIM = 24
ACTION_DIM = 2
ENVIRONMENT_DIM = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one isolated G27 reference or one-step branch."
    )
    parser.add_argument("--mode", choices=("reference", "branch"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--navigation-actor", type=Path, required=True)
    parser.add_argument("--interaction-actor", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path)
    parser.add_argument("--branch", choices=("N1", "N2", "I1", "I2"))
    parser.add_argument(
        "--launchfile", default="multi_robot_scenario_strong_interaction_pilot_5.launch"
    )
    parser.add_argument("--seed", type=int, default=20260910)
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


class FrozenActor:
    def __init__(self, checkpoint):
        self.model = Actor(STATE_DIM, ACTION_DIM)
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def raw_action(self, state):
        values = torch.from_numpy(np.asarray(state, dtype=np.float32)).reshape(1, -1)
        return self.model(values).cpu().numpy().reshape(-1)


class FrozenDetector:
    def __init__(self, checkpoint):
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint, map_location="cpu")
        self.model = LocalRobotDetector(**payload.get("model_config", {}))
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def probabilities(self, patches):
        if len(patches) == 0:
            return np.empty((0,), dtype=np.float32)
        values = torch.from_numpy(np.asarray(patches, dtype=np.float32))
        return torch.sigmoid(self.model(values)[0]).cpu().numpy()


def environment(args):
    os.environ["DRL_MULTI_SCENARIO"] = "manifest"
    os.environ["DRL_MULTI_MANIFEST_PATH"] = str(args.manifest.resolve())
    os.environ["DRL_MULTI_MANIFEST_SAMPLING"] = "cycle"
    os.environ["DRL_MULTI_FIXED_PHYSICS_STEP_SIZE"] = "0.001"
    os.environ["DRL_MULTI_REQUIRE_FIXED_STEP_SERVICE"] = "1"
    os.environ["DRL_MULTI_RECORD_RAW_LIDAR"] = "1" if args.detector_checkpoint else "0"
    os.environ["DRL_MULTI_RAW_LIDAR_VOXEL_SIZE"] = "0.01"
    os.environ["DRL_MULTI_RAW_LIDAR_MAX_RANGE"] = "6.0"
    return MultiAgentGazeboEnv(
        args.launchfile,
        ENVIRONMENT_DIM,
        agent_names=list(AGENT_NAMES),
        cooperative_reward=False,
        anti_stagnation_reward=False,
        safe_recovery_reward=False,
        wall_clearance_reward=False,
        local_navigation_reward=False,
        robot_safe_distance=0.0,
        robot_proximity_penalty_weight=0.0,
        robot_proximity_speed_penalty_weight=0.0,
        robot_clearance_reward_weight=0.0,
        yield_priority_reward=False,
        progress_reward_weight=20.0,
        forward_reward_weight=0.5,
        turn_penalty_weight=0.2,
        obstacle_penalty_weight=0.5,
        stagnation_penalty_weight=0.03,
        weak_coupling_layout=True,
        scenario_mode="manifest",
        fixed_physics_step_size=0.001,
    )


def snapshot(env, states, active_mask):
    agents = {}
    for index, name in enumerate(AGENT_NAMES):
        odom = env.last_odom[name]
        agents[name] = {
            "position": [float(value) for value in env.robot_positions[name]],
            "yaw": float(env._get_robot_yaw(name)),
            "linear_velocity": [
                float(odom.twist.twist.linear.x),
                float(odom.twist.twist.linear.y),
            ],
            "angular_velocity": float(odom.twist.twist.angular.z),
            "actor_state": [float(value) for value in states[index]],
        }
    return {
        "agent_names": list(AGENT_NAMES),
        "active_mask": [bool(value) for value in active_mask],
        "agents": agents,
    }


def actor_actions(states, active_mask, actor):
    raw = []
    deployed = []
    for index, state in enumerate(states):
        if active_mask[index]:
            current_raw = actor.raw_action(state)
            current_deployed = deployed_action(current_raw)
        else:
            current_raw = np.zeros(2, dtype=np.float64)
            current_deployed = np.zeros(2, dtype=np.float64)
        raw.append([float(value) for value in current_raw])
        deployed.append([float(value) for value in current_deployed])
    return raw, deployed


def update_active(active_mask, dones):
    for index, done in enumerate(dones):
        if active_mask[index] and done:
            active_mask[index] = False


def nearest_active_distance(env, ego_index, active_mask):
    ego_name = AGENT_NAMES[ego_index]
    distances = [
        float(
            np.linalg.norm(
                env.robot_positions[name] - env.robot_positions[ego_name]
            )
        )
        for index, name in enumerate(AGENT_NAMES)
        if index != ego_index and active_mask[index]
    ]
    return min(distances) if distances else float("inf")


def gate_feature_at_state(
    env,
    states,
    active_mask,
    ego_index,
    navigation,
    interaction,
    detector,
    tracker,
    logical_time,
):
    ego_name = AGENT_NAMES[ego_index]
    odom = env.last_odom[ego_name]
    pose = np.asarray(
        [
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            env._get_robot_yaw(ego_name),
        ],
        dtype=np.float32,
    )
    examples = build_frame_examples(
        env.raw_lidar_points[ego_name],
        pose,
        [],
        max_background_candidates=12,
    )
    probabilities = detector.probabilities(examples.patches)
    tracked = tracker.update(
        examples.candidate_centers, probabilities, pose, logical_time
    )
    base = build_gate_feature(states[ego_index], tracked, max_tracks=4)
    raw_navigation = navigation.raw_action(states[ego_index])
    raw_interaction = interaction.raw_action(states[ego_index])
    actor = actor_comparison_features(
        raw_navigation.reshape(1, -1), raw_interaction.reshape(1, -1)
    )[0]
    combined = np.concatenate((base, actor)).astype(np.float32)
    if base.shape != (76,) or actor.shape != (6,) or combined.shape != (82,):
        raise RuntimeError("unexpected G27 Gate feature dimensions")
    return base, actor, combined


def reset_case(env, scenario_index, scenario_id):
    env.curriculum_case_index = int(scenario_index)
    states = env.reset()
    actual = str((env.current_curriculum_case or {}).get("scenario_id"))
    if actual != str(scenario_id):
        raise RuntimeError(
            "manifest case mismatch: requested %s but reset %s" % (scenario_id, actual)
        )
    return states, [True] * len(AGENT_NAMES)


def run_reference(args, env, navigation, interaction, detector, candidate):
    scenario_id = str(candidate["scenario_id"])
    scenario_index = int(candidate["scenario_index"])
    anchor_step = int(candidate["anchor_step"])
    ego_index = int(candidate["ego_index"])
    states, active_mask = reset_case(env, scenario_index, scenario_id)
    action_prefix = []
    feature_history = []
    tracker = RobotCandidateTracker() if detector is not None else None
    for step in range(anchor_step):
        if detector is not None and step % 2 == 0 and active_mask[ego_index]:
            feature_history.append(
                gate_feature_at_state(
                    env,
                    states,
                    active_mask,
                    ego_index,
                    navigation,
                    interaction,
                    detector,
                    tracker,
                    logical_time=step * 0.2,
                )
            )
        _, actions = actor_actions(states, active_mask, navigation)
        action_prefix.append(actions)
        states, _, dones, _, _ = env.step(actions, active_mask)
        update_active(active_mask, dones)
    if not active_mask[ego_index]:
        return {
            "format_version": 1,
            "protocol": str(candidate.get("protocol", "G27-P1-fresh-process-one-step-v1")),
            "anchor_id": str(candidate["anchor_id"]),
            "selection_hash": str(candidate["selection_hash"]),
            "scenario_id": scenario_id,
            "scenario_index": scenario_index,
            "anchor_step": anchor_step,
            "ego_index": ego_index,
            "ego_name": AGENT_NAMES[ego_index],
            "valid": False,
            "invalid_reason": "ego_inactive_before_anchor",
            "active_mask": [bool(value) for value in active_mask],
        }

    if detector is not None:
        feature_history.append(
            gate_feature_at_state(
                env,
                states,
                active_mask,
                ego_index,
                navigation,
                interaction,
                detector,
                tracker,
                logical_time=anchor_step * 0.2,
            )
        )

    navigation_raw, navigation_actions = actor_actions(states, active_mask, navigation)
    interaction_raw, all_interaction_actions = actor_actions(
        states, active_mask, interaction
    )
    interaction_actions = [list(action) for action in navigation_actions]
    interaction_actions[ego_index] = list(all_interaction_actions[ego_index])
    distance = nearest_active_distance(env, ego_index, active_mask)
    payload = {
        "format_version": 1,
        "protocol": str(candidate.get("protocol", "G27-P0-fresh-process-one-step-v1")),
        "anchor_id": str(candidate["anchor_id"]),
        "selection_hash": str(candidate["selection_hash"]),
        "scenario_id": scenario_id,
        "scenario_index": scenario_index,
        "anchor_step": anchor_step,
        "ego_index": ego_index,
        "ego_name": AGENT_NAMES[ego_index],
        "agent_names": list(AGENT_NAMES),
        "action_prefix": action_prefix,
        "anchor_snapshot": snapshot(env, states, active_mask),
        "navigation_raw_actions": navigation_raw,
        "interaction_raw_actions": interaction_raw,
        "navigation_actions": navigation_actions,
        "interaction_actions": interaction_actions,
        "reference_nearest_robot_distance": distance,
        "reference_interaction_label": int(distance <= 2.0),
        "reference_action_disagreement_l2": float(
            np.linalg.norm(
                np.asarray(navigation_actions[ego_index])
                - np.asarray(interaction_actions[ego_index])
            )
        ),
        "valid": True,
    }
    if feature_history:
        base_history = np.asarray([item[0] for item in feature_history], dtype=np.float32)
        actor_history = np.asarray([item[1] for item in feature_history], dtype=np.float32)
        combined_history = np.asarray(
            [item[2] for item in feature_history], dtype=np.float32
        )
        combined_padded = np.zeros((8, 82), dtype=np.float32)
        retained = combined_history[-8:]
        combined_padded[-len(retained) :] = retained
        payload["gate_features"] = {
            "sampling_stride_environment_steps": 2,
            "sequence_length": 8,
            "base_feature_history": base_history.tolist(),
            "actor_feature_history": actor_history.tolist(),
            "combined_feature_history": combined_history.tolist(),
            "combined_feature_window": combined_padded.tolist(),
        }
    validate_branch_spec(payload)
    payload["spec_sha256"] = canonical_sha256(payload)
    return payload


def run_branch(args, env, specification):
    validate_branch_spec(specification)
    branch = str(args.branch)
    states, active_mask = reset_case(
        env, specification["scenario_index"], specification["scenario_id"]
    )
    for actions in specification["action_prefix"]:
        states, _, dones, _, _ = env.step(actions, active_mask)
        update_active(active_mask, dones)
    observed = snapshot(env, states, active_mask)
    alignment = alignment_metrics(specification["anchor_snapshot"], observed)
    actions = (
        specification["interaction_actions"]
        if branch.startswith("I")
        else specification["navigation_actions"]
    )
    ego_index = int(specification["ego_index"])
    if not alignment["passed"]:
        payload = {
            "format_version": 1,
            "protocol": specification["protocol"],
            "anchor_id": specification["anchor_id"],
            "spec_sha256": specification["spec_sha256"],
            "branch": branch,
            "actor": "interaction" if branch.startswith("I") else "navigation",
            "alignment": alignment,
            "reward_evaluated": False,
            "reward": None,
            "reward_components": None,
            "target": None,
            "collision": None,
            "action": [float(value) for value in actions[ego_index]],
            "post_step": None,
        }
        payload["result_sha256"] = canonical_sha256(payload)
        return payload
    _, rewards, _, targets, collisions = env.step(actions, active_mask)
    ego_name = AGENT_NAMES[ego_index]
    info = env.last_step_info["agents"][ego_name]
    components = base_reward_components(
        targets[ego_index],
        collisions[ego_index],
        actions[ego_index],
        info["min_laser"],
        info["progress"],
    )
    reward = float(rewards[ego_index])
    if not np.isclose(reward, components["total"], rtol=0.0, atol=1e-6):
        raise RuntimeError(
            "environment/base reward mismatch: %.9f versus %.9f"
            % (reward, components["total"])
        )
    payload = {
        "format_version": 1,
        "protocol": specification["protocol"],
        "anchor_id": specification["anchor_id"],
        "spec_sha256": specification["spec_sha256"],
        "branch": branch,
        "actor": "interaction" if branch.startswith("I") else "navigation",
        "alignment": alignment,
        "reward_evaluated": True,
        "reward": reward,
        "reward_components": components,
        "target": bool(targets[ego_index]),
        "collision": bool(collisions[ego_index]),
        "action": [float(value) for value in actions[ego_index]],
        "post_step": {
            "distance": float(info["distance"]),
            "progress": float(info["progress"]),
            "min_laser": float(info["min_laser"]),
            "nearest_robot_distance": (
                float(info["nearest_robot_distance"])
                if info["nearest_robot_distance"] is not None
                else None
            ),
        },
    }
    payload["result_sha256"] = canonical_sha256(payload)
    return payload


def main():
    args = parse_args()
    if args.mode == "branch" and args.branch is None:
        raise ValueError("--branch is required in branch mode")
    if args.mode == "reference" and args.branch is not None:
        raise ValueError("--branch is only valid in branch mode")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite branch artifact: %s" % args.output)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    navigation = FrozenActor(args.navigation_actor)
    interaction = FrozenActor(args.interaction_actor)
    detector = (
        FrozenDetector(args.detector_checkpoint)
        if args.mode == "reference" and args.detector_checkpoint is not None
        else None
    )
    env = None
    try:
        env = environment(args)
        input_payload = load_json(args.input)
        if args.mode == "reference":
            output = run_reference(
                args, env, navigation, interaction, detector, input_payload
            )
        else:
            output = run_branch(args, env, input_payload)
        write_json(args.output, output)
        print(json.dumps(output, ensure_ascii=False, indent=2))
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
    # rospy/Gazebo bindings can segfault during interpreter teardown after all
    # artifacts and child processes are already closed. Avoid that teardown path.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
