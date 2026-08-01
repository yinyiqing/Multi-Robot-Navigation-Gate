#!/usr/bin/env python3
"""Calibrate a 24D TD3 Critic with controlled same-state Gazebo branches."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from actor_models import Actor
from collect_actor_counterfactuals import (
    ACTION_DIM,
    AGENT_NAMES,
    ENVIRONMENT_DIM,
    STATE_DIM,
    ego_clearance,
    physics_snapshot,
    replay_alignment,
    replay_to_anchor,
    reset_to_case,
    select_anchor_agents,
)
from critic_calibration import (
    discounted_n_step_target,
    summarize_counterfactual_calibration,
)
from critic_models import Critic
from multi_agent_velodyne_env import MultiAgentGazeboEnv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_ACTOR = (
    ROOT
    / "TD3"
    / "pytorch_models"
    / "TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best_actor.pth"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare Critic action rankings with controlled N-step returns while "
            "holding all non-ego agents on the frozen policy."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-actor", type=Path, default=DEFAULT_BASE_ACTOR)
    parser.add_argument(
        "--launchfile", default="multi_robot_scenario_strong_interaction_pilot_5.launch"
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--anchor-stride", type=int, default=4)
    parser.add_argument("--max-anchors-per-episode", type=int, default=4)
    parser.add_argument("--agents-per-anchor", type=int, default=2)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--speeds", default="-1.0,-0.5,0.0,0.5,1.0")
    parser.add_argument("--discount", type=float, default=0.999)
    parser.add_argument("--minimum-observed-gap", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


class FrozenPolicyAndCritic:
    def __init__(self, checkpoint_path, base_actor_path, device):
        self.device = torch.device(device)
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        network = checkpoint["network"]
        self.actor = Actor(STATE_DIM, ACTION_DIM).to(self.device)
        self.actor.load_state_dict(torch.load(base_actor_path, map_location=self.device))
        self.actor.eval()
        self.critic = Critic(STATE_DIM, ACTION_DIM).to(self.device)
        self.critic.load_state_dict(network["critic"])
        self.critic.eval()

    @torch.no_grad()
    def action(self, state):
        value = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        return self.actor(value.reshape(1, -1)).cpu().numpy().reshape(-1)

    @torch.no_grad()
    def q_values(self, state, action):
        state_value = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        action_value = torch.as_tensor(
            action, dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        q1, q2 = self.critic(state_value, action_value)
        return float(q1.item()), float(q2.item())


def policy_actions(states, active_mask, policy):
    raw_actions = []
    env_actions = []
    for index, state in enumerate(states):
        raw = (
            policy.action(state)
            if active_mask[index]
            else np.zeros(ACTION_DIM, dtype=np.float32)
        )
        raw_actions.append(raw)
        env_actions.append([float((raw[0] + 1.0) / 2.0), float(raw[1])])
    return env_actions, raw_actions


def run_speed_branch(
    env,
    states,
    active_mask,
    ego_index,
    speed,
    policy,
    horizon,
    discount,
    initial_raw_actions,
):
    branch_active_mask = list(active_mask)
    reference_active_mask = list(active_mask)
    initial_distance = float(states[ego_index][-4])
    minimum_clearance = ego_clearance(env, ego_index, reference_active_mask)
    rewards = []
    ego_collision = False
    ego_target = False
    final_distance = initial_distance

    for step in range(horizon):
        if step == 0:
            raw_actions = [np.asarray(action).copy() for action in initial_raw_actions]
            raw_actions[ego_index][0] = speed
            env_actions = [
                [float((action[0] + 1.0) / 2.0), float(action[1])]
                for action in raw_actions
            ]
        else:
            env_actions, _ = policy_actions(states, branch_active_mask, policy)
        states, step_rewards, dones, targets, collisions = env.step(
            env_actions, branch_active_mask
        )
        rewards.append(float(step_rewards[ego_index]))
        final_distance = float(states[ego_index][-4])
        minimum_clearance = min(
            minimum_clearance,
            ego_clearance(env, ego_index, reference_active_mask),
        )
        ego_collision = bool(ego_collision or collisions[ego_index])
        ego_target = bool(ego_target or targets[ego_index])
        for index, done in enumerate(dones):
            if done and branch_active_mask[index]:
                branch_active_mask[index] = False
        if not branch_active_mask[ego_index]:
            break

    bootstrap_qmin = 0.0
    if branch_active_mask[ego_index]:
        final_action = policy.action(states[ego_index])
        q1, q2 = policy.q_values(states[ego_index], final_action)
        bootstrap_qmin = min(q1, q2)
    return {
        "observed_rewards": rewards,
        "observed_discounted_reward": discounted_n_step_target(
            rewards, discount, 0.0
        ),
        "bootstrap_qmin": bootstrap_qmin,
        "observed_n_step_target": discounted_n_step_target(
            rewards, discount, bootstrap_qmin
        ),
        "ego_collision": ego_collision,
        "ego_target": ego_target,
        "minimum_ego_clearance": float(minimum_clearance),
        "ego_progress": float(initial_distance - final_distance),
        "steps": len(rewards),
    }


def alignment_is_repeatable(alignment):
    return bool(
        alignment["active_mask_match"]
        and alignment["position_error"] <= 0.02
        and alignment["yaw_error"] <= 0.02
        and alignment["linear_velocity_error"] <= 0.02
        and alignment["angular_velocity_error"] <= 0.03
        and alignment["actor_state_error"] <= 0.05
    )


def main():
    args = parse_args()
    if min(
        args.episodes,
        args.horizon,
        args.anchor_stride,
        args.max_anchors_per_episode,
        args.agents_per_anchor,
    ) < 1:
        raise ValueError("calibration dimensions must be positive")
    if not 0.0 <= args.discount <= 1.0:
        raise ValueError("discount must be in [0, 1]")
    speeds = tuple(float(value) for value in args.speeds.split(","))
    if len(speeds) < 2 or any(value < -1.0 or value > 1.0 for value in speeds):
        raise ValueError("speeds must contain at least two values in [-1, 1]")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ["DRL_MULTI_SCENARIO"] = "manifest"
    os.environ["DRL_MULTI_MANIFEST_PATH"] = str(args.manifest.resolve())
    os.environ["DRL_MULTI_MANIFEST_SAMPLING"] = "cycle"
    policy = FrozenPolicyAndCritic(
        args.checkpoint, args.base_actor, args.device
    )
    env = MultiAgentGazeboEnv(
        args.launchfile,
        ENVIRONMENT_DIM,
        agent_names=list(AGENT_NAMES),
        cooperative_reward=True,
        cooperative_reward_self_weight=0.8,
        cooperative_reward_distance_weighted=True,
        cooperative_reward_sigma=2.0,
        cooperative_reward_mode="average",
        progress_reward_weight=10.0,
        forward_reward_weight=0.0,
        turn_penalty_weight=0.05,
        obstacle_penalty_weight=1.0,
        stagnation_penalty_weight=0.0,
        robot_safe_distance=0.0,
        active_neighbors_only=True,
        weak_coupling_layout=True,
        scenario_mode="manifest",
        fixed_physics_step_size=0.001,
    )
    records = []
    try:
        for episode in range(args.episodes):
            case_index = episode % len(env.curriculum_cases)
            states, active_mask = reset_to_case(env, case_index)
            scenario = env.current_curriculum_case or {}
            scenario_id = str(
                scenario.get("scenario_id") or scenario.get("name") or case_index
            )
            action_prefix = []
            anchor_count = 0
            for step in range(args.max_episode_steps):
                if not any(active_mask):
                    break
                if (
                    all(active_mask)
                    and step % args.anchor_stride == 0
                    and anchor_count < args.max_anchors_per_episode
                ):
                    egos = select_anchor_agents(
                        env, active_mask, args.agents_per_anchor, anchor_count
                    )
                    anchor_states = [np.asarray(value).copy() for value in states]
                    anchor_active_mask = list(active_mask)
                    snapshot = physics_snapshot(env, anchor_states)
                    _, anchor_raw_actions = policy_actions(
                        anchor_states, anchor_active_mask, policy
                    )
                    for ego_index in egos:
                        actor_action = anchor_raw_actions[ego_index]
                        for speed in speeds:
                            branch_states, branch_active_mask = replay_to_anchor(
                                env, case_index, action_prefix
                            )
                            alignment = replay_alignment(
                                env,
                                snapshot,
                                anchor_active_mask,
                                branch_states,
                                branch_active_mask,
                            )
                            candidate_action = actor_action.copy()
                            candidate_action[0] = speed
                            q1, q2 = policy.q_values(
                                anchor_states[ego_index], candidate_action
                            )
                            outcome = run_speed_branch(
                                env,
                                branch_states,
                                branch_active_mask,
                                ego_index,
                                speed,
                                policy,
                                args.horizon,
                                args.discount,
                                anchor_raw_actions,
                            )
                            records.append(
                                {
                                    "scenario_id": scenario_id,
                                    "anchor_step": step,
                                    "ego_index": ego_index,
                                    "raw_linear_speed": speed,
                                    "raw_angular_action": float(actor_action[1]),
                                    "predicted_q1": q1,
                                    "predicted_q2": q2,
                                    "predicted_qmin": min(q1, q2),
                                    "repeatable": alignment_is_repeatable(alignment),
                                    "alignment": alignment,
                                    **outcome,
                                }
                            )
                    states, active_mask = replay_to_anchor(
                        env, case_index, action_prefix
                    )
                    anchor_count += 1

                env_actions, _ = policy_actions(states, active_mask, policy)
                action_prefix.append(env_actions)
                states, _, dones, _, _ = env.step(env_actions, active_mask)
                for index, done in enumerate(dones):
                    if done and active_mask[index]:
                        active_mask[index] = False
            print(
                "episode=%d scenario=%s anchors=%d records=%d"
                % (episode + 1, scenario_id, anchor_count, len(records)),
                flush=True,
            )
    finally:
        env.close()

    result = {
        "format_version": 1,
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint),
        "base_actor": str(args.base_actor),
        "speeds": speeds,
        "horizon": args.horizon,
        "discount": args.discount,
        "summary": summarize_counterfactual_calibration(
            records, args.minimum_observed_gap
        ),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
