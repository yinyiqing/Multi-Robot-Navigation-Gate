#!/usr/bin/env python3
"""Audit action support and local-Critic preferences from a training checkpoint."""

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


SPEEDS = (-1.0, -0.5, 0.0, 0.5, 1.0)
ACTION_BIN_EDGES = (-1.000001, -0.6, -0.2, 0.2, 0.6, 1.000001)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--base-actor", type=Path)
    parser.add_argument("--context-feature-dim", type=int, default=7)
    parser.add_argument("--close-distance", type=float, default=1.2)
    parser.add_argument("--min-closing-speed", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def optimizer_steps(state):
    values = state.get("state", {}).values()
    if not values:
        return 0
    step = next(iter(values)).get("step", 0)
    return int(step.item() if torch.is_tensor(step) else step)


def relative_parameter_l2(candidate, reference):
    delta_squared = 0.0
    reference_squared = 0.0
    for name, reference_parameter in reference.items():
        candidate_parameter = candidate[name]
        delta_squared += float(torch.sum((candidate_parameter - reference_parameter).float() ** 2))
        reference_squared += float(torch.sum(reference_parameter.float() ** 2))
    return float(np.sqrt(delta_squared / max(reference_squared, 1e-12)))


def context_masks(critic_states, actor_state_dim, feature_dim, close_distance, min_closing_speed):
    contexts = critic_states[:, actor_state_dim:]
    if contexts.shape[1] == 0 or contexts.shape[1] % feature_dim:
        raise ValueError("Critic context is incompatible with context-feature-dim")
    if feature_dim < 7:
        raise ValueError("Approaching-state audit requires ego-motion context")
    slots = contexts.reshape(len(contexts), -1, feature_dim)
    valid = slots[:, :, feature_dim - 1] > 0.5
    distances = np.where(valid, slots[:, :, 2], np.inf)
    radial_velocity = np.sum(slots[:, :, :2] * slots[:, :, 4:6], axis=2) / np.maximum(
        slots[:, :, 2], 1e-6
    )
    closing = valid & (distances <= close_distance) & (-radial_velocity >= min_closing_speed)
    return {
        "all": np.ones(len(critic_states), dtype=bool),
        "interaction": np.min(distances, axis=1) <= 2.0,
        "close": np.min(distances, axis=1) <= close_distance,
        "close_approaching": np.any(closing, axis=1),
    }


def evaluate_model(actor, critic, actor_states, critic_states, masks, batch_size):
    actions = []
    q1_by_speed = [[] for _ in SPEEDS]
    q2_by_speed = [[] for _ in SPEEDS]
    gradients = []
    for start in range(0, len(actor_states), batch_size):
        actor_state = torch.from_numpy(actor_states[start : start + batch_size])
        critic_state = torch.from_numpy(critic_states[start : start + batch_size])
        with torch.no_grad():
            actor_action = actor(actor_state)
            actions.append(actor_action.numpy())
            for index, speed in enumerate(SPEEDS):
                swept = actor_action.clone()
                swept[:, 0] = speed
                q1, q2 = critic(critic_state, swept)
                q1_by_speed[index].append(q1.numpy().ravel())
                q2_by_speed[index].append(q2.numpy().ravel())
        actor_action = actor_action.detach().requires_grad_(True)
        q1, _ = critic(critic_state, actor_action)
        gradients.append(torch.autograd.grad(q1.sum(), actor_action)[0].numpy())

    actor_actions = np.concatenate(actions)
    q1 = np.stack([np.concatenate(parts) for parts in q1_by_speed], axis=1)
    q2 = np.stack([np.concatenate(parts) for parts in q2_by_speed], axis=1)
    qmin = np.minimum(q1, q2)
    action_gradients = np.concatenate(gradients)
    result = {}
    for name, mask in masks.items():
        count = int(mask.sum())
        if not count:
            result[name] = {"count": 0}
            continue
        selected_qmin = qmin[mask]
        selected_gradients = action_gradients[mask]
        angular_positive = float((selected_gradients[:, 1] > 0.0).mean())
        result[name] = {
            "count": count,
            "qmin_mean_by_raw_linear": [float(value) for value in selected_qmin.mean(axis=0)],
            "full_best_share": float((np.argmax(selected_qmin, axis=1) == len(SPEEDS) - 1).mean()),
            "full_minus_stop_mean": float((selected_qmin[:, -1] - selected_qmin[:, 0]).mean()),
            "twin_abs_disagreement_mean": float(np.abs(q1[mask] - q2[mask]).mean()),
            "linear_gradient_positive_share": float((selected_gradients[:, 0] > 0.0).mean()),
            "linear_gradient_mean": float(selected_gradients[:, 0].mean()),
            "angular_gradient_one_sided_share": max(angular_positive, 1.0 - angular_positive),
        }
    return actor_actions, result


def action_bin_diagnostics(actions, rewards, dones, masks):
    result = {}
    negative_terminal = dones & (rewards < 0.0)
    positive_terminal = dones & (rewards > 0.0)
    for group_name, group_mask in masks.items():
        bins = []
        for lower, upper in zip(ACTION_BIN_EDGES[:-1], ACTION_BIN_EDGES[1:]):
            mask = group_mask & (actions[:, 0] >= lower) & (actions[:, 0] < upper)
            count = int(mask.sum())
            bins.append(
                {
                    "raw_linear_range": [float(max(lower, -1.0)), float(min(upper, 1.0))],
                    "count": count,
                    "reward_mean": float(rewards[mask].mean()) if count else None,
                    "negative_terminal_count": int(np.sum(mask & negative_terminal)),
                    "positive_terminal_count": int(np.sum(mask & positive_terminal)),
                }
            )
        result[group_name] = bins
    return result


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    replay = checkpoint["replay_buffer"]["buffer"]
    if not replay or len(replay[0]) not in (7, 8):
        raise ValueError("Checkpoint does not contain local-Critic replay transitions")

    actor_states = np.stack([item[0] for item in replay]).astype(np.float32)
    critic_states = np.stack([item[1] for item in replay]).astype(np.float32)
    behavior_actions = np.stack([item[2] for item in replay]).astype(np.float32)
    rewards = np.asarray([item[3] for item in replay], dtype=np.float32)
    dones = np.asarray([bool(item[4]) for item in replay], dtype=bool)
    network = checkpoint["network"]

    actor = Actor(actor_states.shape[1], behavior_actions.shape[1])
    actor.load_state_dict(network["actor"])
    actor.eval()
    critic = Critic(critic_states.shape[1], behavior_actions.shape[1])
    critic.load_state_dict(network["critic"])
    critic.eval()
    masks = context_masks(
        critic_states,
        actor_states.shape[1],
        args.context_feature_dim,
        args.close_distance,
        args.min_closing_speed,
    )
    actor_actions, preferences = evaluate_model(
        actor, critic, actor_states, critic_states, masks, args.batch_size
    )

    result = {
        "checkpoint": str(args.checkpoint),
        "agent_samples": int(checkpoint.get("timestep", len(replay))),
        "replay_states": len(replay),
        "behavior_action_coverage": {
            "raw_linear_mean": float(behavior_actions[:, 0].mean()),
            "raw_linear_std": float(behavior_actions[:, 0].std()),
            "raw_linear_le_neg_0p8_share": float((behavior_actions[:, 0] <= -0.8).mean()),
            "raw_linear_mid_share": float((np.abs(behavior_actions[:, 0]) < 0.8).mean()),
            "raw_linear_ge_0p8_share": float((behavior_actions[:, 0] >= 0.8).mean()),
            "angular_delta_from_actor_mean_absolute": float(
                np.abs(behavior_actions[:, 1] - actor_actions[:, 1]).mean()
            ),
        },
        "reward": {
            "done_rate": float(dones.mean()),
            "negative_terminal_count": int(np.sum(dones & (rewards < 0.0))),
            "positive_terminal_count": int(np.sum(dones & (rewards > 0.0))),
            "negative_terminal_rate": float(np.mean(dones & (rewards < 0.0))),
        },
        "state_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "behavior_by_raw_linear_bin": action_bin_diagnostics(
            behavior_actions,
            rewards,
            dones,
            {
                "all": masks["all"],
                "close_approaching": masks["close_approaching"],
            },
        ),
        "preferences": preferences,
        "critic_optimizer_steps": optimizer_steps(network["critic_optimizer"]),
        "actor_optimizer_steps": optimizer_steps(network["actor_optimizer"]),
    }
    if args.base_actor:
        base_actor = torch.load(args.base_actor, map_location="cpu", weights_only=False)
        result["actor_relative_parameter_l2_from_base"] = relative_parameter_l2(
            network["actor"], base_actor
        )

    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
