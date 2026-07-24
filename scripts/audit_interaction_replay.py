#!/usr/bin/env python3
"""Audit how interaction states are represented in a local-Critic replay buffer."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "TD3"))

from actor_models import Actor  # noqa: E402
from critic_models import Critic  # noqa: E402


BANDS = (
    ("critical_le_0p8", 0.0, 0.8),
    ("near_0p8_to_1p2", 0.8, 1.2),
    ("interaction_1p2_to_2p0", 1.2, 2.0),
    ("visible_far_gt_2p0", 2.0, float("inf")),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure interaction-state representation in saved replay data."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--base-actor", type=Path)
    parser.add_argument("--candidate-actor", type=Path)
    parser.add_argument("--critic", type=Path)
    parser.add_argument("--context-feature-dim", type=int, default=5)
    parser.add_argument("--change-threshold", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def interaction_band_masks(critic_states, state_dim, feature_dim):
    context = critic_states[:, state_dim:]
    if context.shape[1] == 0 or context.shape[1] % feature_dim:
        raise ValueError("Critic context is incompatible with context-feature-dim")
    slots = context.reshape(len(context), -1, feature_dim)
    valid = slots[:, :, feature_dim - 1] > 0.5
    distances = np.where(valid, slots[:, :, 2], np.inf)
    nearest = np.min(distances, axis=1)
    masks = {"no_visible_neighbor": ~np.isfinite(nearest)}
    for name, lower, upper in BANDS:
        masks[name] = np.isfinite(nearest) & (nearest > lower) & (nearest <= upper)
    return nearest, valid.sum(axis=1), masks


def vector(values):
    return [float(value) for value in values]


def summarize_group(mask, rewards, dones, actions, nearest, neighbor_counts):
    count = int(mask.sum())
    if not count:
        return {"count": 0, "share": 0.0}
    group_rewards = rewards[mask]
    group_actions = actions[mask]
    group_nearest = nearest[mask]
    finite_nearest = group_nearest[np.isfinite(group_nearest)]
    return {
        "count": count,
        "share": float(mask.mean()),
        "done_rate": float(dones[mask].mean()),
        "negative_terminal_rate": float(
            np.mean(dones[mask] & (group_rewards < 0.0))
        ),
        "positive_terminal_rate": float(
            np.mean(dones[mask] & (group_rewards > 0.0))
        ),
        "reward_mean": float(group_rewards.mean()),
        "reward_quantiles": vector(
            np.quantile(group_rewards, [0.0, 0.25, 0.5, 0.75, 1.0])
        ),
        "behavior_raw_action_mean": vector(group_actions.mean(axis=0)),
        "behavior_env_linear_mean": float(((group_actions[:, 0] + 1.0) / 2.0).mean()),
        "visible_neighbor_count_mean": float(neighbor_counts[mask].mean()),
        "nearest_neighbor_distance_mean": (
            float(finite_nearest.mean()) if finite_nearest.size else None
        ),
    }


def load_state_dict(model, path):
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    model.eval()
    return model


def add_model_diagnostics(
    result,
    masks,
    actor_states,
    critic_states,
    base_actor_path,
    candidate_actor_path,
    critic_path,
    action_dim,
    change_threshold,
    batch_size,
):
    state_dim = actor_states.shape[1]
    critic_state_dim = critic_states.shape[1]
    base_actor = load_state_dict(Actor(state_dim, action_dim), base_actor_path)
    candidate_actor = load_state_dict(
        Actor(state_dim, action_dim), candidate_actor_path
    )
    critic = load_state_dict(Critic(critic_state_dim, action_dim), critic_path)

    base_actions = []
    candidate_actions = []
    base_q = []
    candidate_q = []
    with torch.no_grad():
        for start in range(0, len(actor_states), batch_size):
            stop = start + batch_size
            actor_state = torch.from_numpy(actor_states[start:stop])
            critic_state = torch.from_numpy(critic_states[start:stop])
            base_action = base_actor(actor_state)
            candidate_action = candidate_actor(actor_state)
            base_value, _ = critic(critic_state, base_action)
            candidate_value, _ = critic(critic_state, candidate_action)
            base_actions.append(base_action.numpy())
            candidate_actions.append(candidate_action.numpy())
            base_q.append(base_value.numpy().ravel())
            candidate_q.append(candidate_value.numpy().ravel())

    base_actions = np.concatenate(base_actions)
    candidate_actions = np.concatenate(candidate_actions)
    base_q = np.concatenate(base_q)
    candidate_q = np.concatenate(candidate_q)
    action_delta = candidate_actions - base_actions
    q_delta = candidate_q - base_q
    changed = np.max(np.abs(action_delta), axis=1) > change_threshold

    diagnostics = {}
    for name, mask in masks.items():
        count = int(mask.sum())
        if not count:
            continue
        group_changed = changed[mask]
        changed_q = q_delta[mask][group_changed]
        diagnostics[name] = {
            "count": count,
            "base_action_mean": vector(base_actions[mask].mean(axis=0)),
            "candidate_action_mean": vector(candidate_actions[mask].mean(axis=0)),
            "action_delta_mean": vector(action_delta[mask].mean(axis=0)),
            "action_delta_mean_absolute": vector(
                np.abs(action_delta[mask]).mean(axis=0)
            ),
            "changed_state_share": float(group_changed.mean()),
            "critic_candidate_minus_base_mean": float(q_delta[mask].mean()),
            "critic_candidate_preferred_share": float((q_delta[mask] > 0.0).mean()),
            "changed_states_candidate_minus_base_mean": (
                float(changed_q.mean()) if changed_q.size else None
            ),
            "changed_states_candidate_preferred_share": (
                float((changed_q > 0.0).mean()) if changed_q.size else None
            ),
        }
    result["model_diagnostics"] = diagnostics


def main():
    args = parse_args()
    model_paths = (args.base_actor, args.candidate_actor, args.critic)
    if any(model_paths) and not all(model_paths):
        raise ValueError(
            "--base-actor, --candidate-actor, and --critic must be used together"
        )
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    replay = checkpoint["replay_buffer"]["buffer"]
    if not replay or len(replay[0]) != 7:
        raise ValueError("Checkpoint does not contain local-Critic replay transitions")

    actor_states = np.stack([item[0] for item in replay]).astype(np.float32)
    critic_states = np.stack([item[1] for item in replay]).astype(np.float32)
    actions = np.stack([item[2] for item in replay]).astype(np.float32)
    rewards = np.asarray([item[3] for item in replay], dtype=np.float32)
    dones = np.asarray([bool(item[4]) for item in replay], dtype=bool)
    state_dim = actor_states.shape[1]
    nearest, neighbor_counts, masks = interaction_band_masks(
        critic_states, state_dim, args.context_feature_dim
    )

    result = {
        "checkpoint": str(args.checkpoint),
        "replay_states": len(replay),
        "state_dim": state_dim,
        "critic_state_dim": int(critic_states.shape[1]),
        "context_feature_dim": args.context_feature_dim,
        "bands": {
            name: summarize_group(
                mask, rewards, dones, actions, nearest, neighbor_counts
            )
            for name, mask in masks.items()
        },
    }
    if all(model_paths):
        add_model_diagnostics(
            result,
            masks,
            actor_states,
            critic_states,
            args.base_actor,
            args.candidate_actor,
            args.critic,
            actions.shape[1],
            args.change_threshold,
            args.batch_size,
        )

    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
