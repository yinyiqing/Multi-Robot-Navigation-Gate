#!/usr/bin/env python3
"""Audit action coverage and speed preference in an original 24D TD3 Critic."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TD3_DIR = PROJECT_ROOT / "TD3"
DEFAULT_BASE_ACTOR = (
    TD3_DIR
    / "pytorch_models"
    / "TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best_actor.pth"
)
sys.path.insert(0, str(TD3_DIR))

from actor_models import Actor  # noqa: E402
from critic_models import Critic  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--base-actor", type=Path, default=DEFAULT_BASE_ACTOR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args()


def load_actor(state_dim, action_dim, state_dict):
    model = Actor(state_dim, action_dim)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_actions(actor, critic, states, batch_size):
    actor_actions = []
    q1_by_speed = {speed: [] for speed in (-1.0, -0.5, 0.0, 0.5, 1.0)}
    qmin_by_speed = {speed: [] for speed in q1_by_speed}
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            state = torch.from_numpy(states[start : start + batch_size])
            actions = actor(state)
            actor_actions.append(actions.numpy())
            for speed in q1_by_speed:
                swept = actions.clone()
                swept[:, 0] = speed
                q1, q2 = critic(state, swept)
                q1_by_speed[speed].append(q1.numpy().ravel())
                qmin_by_speed[speed].append(torch.minimum(q1, q2).numpy().ravel())
    return (
        np.concatenate(actor_actions),
        {key: np.concatenate(value) for key, value in q1_by_speed.items()},
        {key: np.concatenate(value) for key, value in qmin_by_speed.items()},
    )


def summarize_q(mask, q1_by_speed, qmin_by_speed):
    count = int(mask.sum())
    if not count:
        return {"count": 0}
    q1 = {str(speed): float(values[mask].mean()) for speed, values in q1_by_speed.items()}
    qmin = {
        str(speed): float(values[mask].mean())
        for speed, values in qmin_by_speed.items()
    }
    q1_delta = q1_by_speed[1.0][mask] - q1_by_speed[-1.0][mask]
    return {
        "count": count,
        "q1_by_raw_linear": q1,
        "qmin_by_raw_linear": qmin,
        "q1_full_minus_stop_mean": float(q1_delta.mean()),
        "q1_full_preferred_share": float((q1_delta > 0.0).mean()),
    }


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    network = checkpoint["network"]
    replay = checkpoint["replay_buffer"]["buffer"]
    if not replay or len(replay[0]) != 5:
        raise ValueError("Checkpoint does not contain original-Critic replay transitions")

    states = np.stack([item[0] for item in replay]).astype(np.float32)
    behavior_actions = np.stack([item[1] for item in replay]).astype(np.float32)
    rewards = np.asarray([item[2] for item in replay], dtype=np.float32)
    dones = np.asarray([bool(item[3]) for item in replay], dtype=bool)
    state_dim = states.shape[1]
    action_dim = behavior_actions.shape[1]

    base_state = torch.load(args.base_actor, map_location="cpu", weights_only=False)
    base_actor = load_actor(state_dim, action_dim, base_state)
    actor = load_actor(state_dim, action_dim, network["actor"])
    critic = Critic(state_dim, action_dim)
    critic.load_state_dict(network["critic"])
    critic.eval()

    actor_actions, q1_by_speed, qmin_by_speed = evaluate_actions(
        actor, critic, states, args.batch_size
    )
    with torch.no_grad():
        base_actions = base_actor(torch.from_numpy(states)).numpy()

    minimum_laser = states[:, :20].min(axis=1)
    masks = {
        "all": np.ones(len(states), dtype=bool),
        "laser_le_0p5": minimum_laser <= 0.5,
        "laser_le_0p8": minimum_laser <= 0.8,
        "laser_le_1p0": minimum_laser <= 1.0,
    }
    parameter_delta_squared = 0.0
    parameter_base_squared = 0.0
    for name, base_parameter in base_state.items():
        delta = network["actor"][name] - base_parameter
        parameter_delta_squared += float(torch.sum(delta.float() ** 2))
        parameter_base_squared += float(torch.sum(base_parameter.float() ** 2))

    result = {
        "checkpoint": str(args.checkpoint),
        "replay_states": len(replay),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "agent_samples": checkpoint.get("timestep"),
        "critic_optimizer_steps": int(
            next(iter(network["critic_optimizer"]["state"].values()))["step"]
        ),
        "actor_optimizer_steps": (
            int(next(iter(network["actor_optimizer"]["state"].values()))["step"])
            if network["actor_optimizer"]["state"]
            else 0
        ),
        "actor_relative_parameter_l2_from_5a": float(
            np.sqrt(parameter_delta_squared / max(parameter_base_squared, 1e-12))
        ),
        "actor_action_delta_mean_from_5a": [
            float(value) for value in (actor_actions - base_actions).mean(axis=0)
        ],
        "behavior_action_coverage": {
            "raw_linear_near_stop_share": float((behavior_actions[:, 0] <= -0.95).mean()),
            "raw_linear_mid_share": float((np.abs(behavior_actions[:, 0]) < 0.8).mean()),
            "raw_linear_near_full_share": float((behavior_actions[:, 0] >= 0.95).mean()),
        },
        "reward": {
            "mean": float(rewards.mean()),
            "terminal_rate": float(dones.mean()),
            "negative_terminal_rate": float((dones & (rewards < 0.0)).mean()),
            "positive_terminal_rate": float((dones & (rewards > 0.0)).mean()),
        },
        "speed_preference": {
            name: summarize_q(mask, q1_by_speed, qmin_by_speed)
            for name, mask in masks.items()
        },
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
