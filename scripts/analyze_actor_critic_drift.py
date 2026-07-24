#!/usr/bin/env python3
"""Audit Actor drift and Critic preference on a saved local-Critic replay buffer."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TD3_DIR = PROJECT_ROOT / "TD3"
sys.path.insert(0, str(TD3_DIR))

from actor_models import Actor  # noqa: E402
from critic_models import Critic  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-actor", required=True, type=Path)
    parser.add_argument("--candidate-actor", required=True, type=Path)
    parser.add_argument("--critic", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--state-dim", type=int, default=24)
    parser.add_argument("--critic-state-dim", type=int, default=69)
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--change-threshold", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args()


def load_model(model_type, path, *dimensions):
    model = model_type(*dimensions)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    model.eval()
    return model


def vector(values):
    return [float(value) for value in values]


def main():
    args = parse_args()
    base_actor = load_model(
        Actor, args.base_actor, args.state_dim, args.action_dim
    )
    candidate_actor = load_model(
        Actor, args.candidate_actor, args.state_dim, args.action_dim
    )
    critic = load_model(
        Critic, args.critic, args.critic_state_dim, args.action_dim
    )

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    replay = checkpoint["replay_buffer"]["buffer"]
    actor_states = np.stack([transition[0] for transition in replay]).astype(
        np.float32
    )
    critic_states = np.stack([transition[1] for transition in replay]).astype(
        np.float32
    )
    behavior_actions = np.stack([transition[2] for transition in replay]).astype(
        np.float32
    )

    base_batches = []
    candidate_batches = []
    base_q_batches = []
    candidate_q_batches = []
    with torch.no_grad():
        for start in range(0, len(replay), args.batch_size):
            stop = start + args.batch_size
            actor_state = torch.from_numpy(actor_states[start:stop])
            critic_state = torch.from_numpy(critic_states[start:stop])
            base_action = base_actor(actor_state)
            candidate_action = candidate_actor(actor_state)
            base_q, _ = critic(critic_state, base_action)
            candidate_q, _ = critic(critic_state, candidate_action)
            base_batches.append(base_action.numpy())
            candidate_batches.append(candidate_action.numpy())
            base_q_batches.append(base_q.numpy().ravel())
            candidate_q_batches.append(candidate_q.numpy().ravel())

    base_actions = np.concatenate(base_batches)
    candidate_actions = np.concatenate(candidate_batches)
    base_q = np.concatenate(base_q_batches)
    candidate_q = np.concatenate(candidate_q_batches)
    delta = candidate_actions - base_actions
    q_delta = candidate_q - base_q
    changed = np.max(np.abs(delta), axis=1) > args.change_threshold

    parameter_drift = {}
    for name, base_parameter in base_actor.state_dict().items():
        candidate_parameter = candidate_actor.state_dict()[name]
        difference = candidate_parameter - base_parameter
        relative_l2 = torch.linalg.vector_norm(difference) / (
            torch.linalg.vector_norm(base_parameter) + 1e-12
        )
        parameter_drift[name] = {
            "relative_l2": float(relative_l2),
            "max_absolute": float(torch.max(torch.abs(difference))),
        }

    changed_q_delta = q_delta[changed]
    result = {
        "replay_states": len(replay),
        "change_threshold": args.change_threshold,
        "base_action_mean": vector(base_actions.mean(axis=0)),
        "candidate_action_mean": vector(candidate_actions.mean(axis=0)),
        "action_delta_mean": vector(delta.mean(axis=0)),
        "action_delta_mean_absolute": vector(np.abs(delta).mean(axis=0)),
        "changed_state_share": float(changed.mean()),
        "sign_flip_share": vector((base_actions * candidate_actions < 0).mean(axis=0)),
        "base_behavior_action_mae": vector(
            np.abs(base_actions - behavior_actions).mean(axis=0)
        ),
        "candidate_behavior_action_mae": vector(
            np.abs(candidate_actions - behavior_actions).mean(axis=0)
        ),
        "critic_q": {
            "base_mean": float(base_q.mean()),
            "candidate_mean": float(candidate_q.mean()),
            "candidate_minus_base_mean": float(q_delta.mean()),
            "candidate_preferred_share": float((q_delta > 0).mean()),
            "changed_states_candidate_minus_base_mean": float(
                changed_q_delta.mean()
            ),
            "changed_states_candidate_preferred_share": float(
                (changed_q_delta > 0).mean()
            ),
        },
        "parameter_drift": parameter_drift,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
