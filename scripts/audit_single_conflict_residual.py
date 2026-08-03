#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "TD3"))

from actor_models import Actor
from residual_teacher_audit import (
    VectorScaledResidual,
    action_error_metrics,
    balanced_class_weights,
    calibrate_residual_scale,
    interaction_labels_from_critic_states,
    teacher_choice_accuracy,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether a small 24D Residual can reproduce the epoch-16 "
            "single-conflict correction without changing normal 5A actions."
        )
    )
    parser.add_argument("--replay-checkpoint", type=Path, required=True)
    parser.add_argument("--generalist-actor", type=Path, required=True)
    parser.add_argument("--specialist-actor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-dim", type=int, default=24)
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--context-feature-dim", type=int, default=7)
    parser.add_argument("--interaction-distance", type=float, default=2.0)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--scale-quantile", type=float, default=0.99)
    parser.add_argument("--scale-maximum", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_actor(path, state_dim, action_dim, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    elif isinstance(payload, dict) and "actor" in payload:
        payload = payload["actor"]
    actor = Actor(state_dim, action_dim).to(device)
    actor.load_state_dict(payload)
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad = False
    return actor


@torch.no_grad()
def actor_actions(actor, states, batch_size, device):
    outputs = []
    for start in range(0, len(states), batch_size):
        batch = torch.from_numpy(states[start : start + batch_size]).to(device)
        outputs.append(actor(batch).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32, copy=False)


@torch.no_grad()
def residual_actions(model, states, batch_size, device):
    model.eval()
    outputs = []
    for start in range(0, len(states), batch_size):
        batch = torch.from_numpy(states[start : start + batch_size]).to(device)
        outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32, copy=False)


def summarize_split(predicted, target, labels, generalist, specialist):
    normal = ~labels
    interaction = labels
    zero = np.zeros_like(target)
    zero_interaction_mse = action_error_metrics(
        zero[interaction], target[interaction]
    )["mse"]
    interaction_metrics = action_error_metrics(
        predicted[interaction], target[interaction]
    )
    improvement = (
        1.0 - interaction_metrics["mse"] / zero_interaction_mse
        if zero_interaction_mse > 0.0
        else 0.0
    )
    return {
        "frames": int(len(labels)),
        "interaction_frames": int(np.sum(interaction)),
        "interaction_rate": float(np.mean(interaction)),
        "all": action_error_metrics(predicted, target),
        "normal": action_error_metrics(predicted[normal], target[normal]),
        "interaction": interaction_metrics,
        "zero_residual_interaction_mse": zero_interaction_mse,
        "interaction_mse_improvement_over_zero": float(improvement),
        "teacher_choice": teacher_choice_accuracy(
            predicted, generalist, specialist, labels
        ),
    }


def main():
    args = parse_args()
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if min(args.epochs, args.batch_size, args.hidden_dim) < 1:
        raise ValueError("epochs, batch size and hidden dim must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(
        args.replay_checkpoint, map_location="cpu", weights_only=False
    )
    replay_state = checkpoint.get("replay_buffer")
    if not isinstance(replay_state, dict) or "buffer" not in replay_state:
        raise ValueError("checkpoint does not contain a replay buffer")
    if replay_state.get("count", 0) >= replay_state.get("buffer_size", 0):
        raise ValueError("wrapped replay order cannot support a temporal split")
    replay = replay_state["buffer"]
    states = np.stack([entry[0] for entry in replay]).astype(np.float32)
    critic_states = np.stack([entry[1] for entry in replay]).astype(np.float32)
    labels = interaction_labels_from_critic_states(
        critic_states,
        args.state_dim,
        args.context_feature_dim,
        args.interaction_distance,
    )
    del critic_states
    split = int(len(states) * args.train_fraction)
    if split < 1 or split >= len(states):
        raise ValueError("temporal split produced an empty partition")

    generalist = load_actor(
        args.generalist_actor, args.state_dim, args.action_dim, device
    )
    specialist = load_actor(
        args.specialist_actor, args.state_dim, args.action_dim, device
    )
    generalist_actions = actor_actions(
        generalist, states, args.batch_size * 4, device
    )
    specialist_actions = actor_actions(
        specialist, states, args.batch_size * 4, device
    )
    teacher_delta = specialist_actions - generalist_actions
    target = np.where(labels[:, None], teacher_delta, 0.0).astype(np.float32)
    train_slice = slice(0, split)
    validation_slice = slice(split, len(states))
    scale = calibrate_residual_scale(
        teacher_delta[train_slice][labels[train_slice]],
        quantile=args.scale_quantile,
        maximum=args.scale_maximum,
    )
    coverage = np.mean(
        np.abs(teacher_delta[validation_slice][labels[validation_slice]])
        <= scale[None, :],
        axis=0,
    )

    model = VectorScaledResidual(
        args.state_dim, args.action_dim, args.hidden_dim, scale
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    weights = balanced_class_weights(labels[train_slice])
    normalized_scale = torch.from_numpy(scale).to(device)
    rng = np.random.default_rng(args.seed)
    best = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(split)
        losses = []
        for start in range(0, split, args.batch_size):
            indices = order[start : start + args.batch_size]
            batch_states = torch.from_numpy(states[indices]).to(device)
            batch_target = torch.from_numpy(target[indices]).to(device)
            batch_weights = torch.from_numpy(weights[indices]).to(device)
            normalized_error = (
                model(batch_states) - batch_target
            ) / normalized_scale
            per_frame = torch.mean(torch.square(normalized_error), dim=1)
            loss = torch.mean(per_frame * batch_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        predicted = residual_actions(
            model,
            states[validation_slice],
            args.batch_size * 4,
            device,
        )
        metrics = summarize_split(
            predicted,
            target[validation_slice],
            labels[validation_slice],
            generalist_actions[validation_slice],
            specialist_actions[validation_slice],
        )
        key = max(
            metrics["normal"]["mse"] / float(np.mean(np.square(scale))),
            metrics["interaction"]["mse"]
            / max(metrics["zero_residual_interaction_mse"], 1e-12),
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "selection_key": float(key),
                "validation": metrics,
            }
        )
        if best is None or key < best[0]:
            best = (key, epoch, copy.deepcopy(model.state_dict()), metrics)
        print(
            "epoch=%d loss=%.6f normal_mae=%.4f interaction_mse_gain=%.3f "
            "interaction_choice=%.3f"
            % (
                epoch,
                float(np.mean(losses)),
                metrics["normal"]["mae"],
                metrics["interaction_mse_improvement_over_zero"],
                metrics["teacher_choice"]["interaction"]["accuracy"] or 0.0,
            )
        )

    model.load_state_dict(best[2])
    train_prediction = residual_actions(
        model, states[train_slice], args.batch_size * 4, device
    )
    validation_prediction = residual_actions(
        model, states[validation_slice], args.batch_size * 4, device
    )
    train_metrics = summarize_split(
        train_prediction,
        target[train_slice],
        labels[train_slice],
        generalist_actions[train_slice],
        specialist_actions[train_slice],
    )
    validation_metrics = summarize_split(
        validation_prediction,
        target[validation_slice],
        labels[validation_slice],
        generalist_actions[validation_slice],
        specialist_actions[validation_slice],
    )
    criteria = {
        "interaction_mse_improvement_at_least_0_40": (
            validation_metrics["interaction_mse_improvement_over_zero"] >= 0.40
        ),
        "normal_per_action_mae_at_most_0_03": (
            max(validation_metrics["normal"]["per_action_mae"]) <= 0.03
        ),
        "interaction_teacher_choice_at_least_0_70": (
            (
                validation_metrics["teacher_choice"]["interaction"]["accuracy"]
                or 0.0
            )
            >= 0.70
        ),
        "validation_scale_coverage_at_least_0_95": bool(
            np.all(coverage >= 0.95)
        ),
    }
    summary = {
        "protocol": "single-conflict-residual-r0-replay-temporal-v1",
        "decision": "pass" if all(criteria.values()) else "fail",
        "criteria": criteria,
        "limitations": [
            "Replay contains exact-edge-1 training scenarios only, but scenario IDs are not stored per transition.",
            "The 80/20 temporal split is not a scenario-level independent validation split.",
            "Offline action fidelity does not establish closed-loop navigation performance.",
        ],
        "config": {
            **vars(args),
            "replay_checkpoint": str(args.replay_checkpoint),
            "generalist_actor": str(args.generalist_actor),
            "specialist_actor": str(args.specialist_actor),
            "output_dir": str(args.output_dir),
            "device": device,
        },
        "artifacts": {
            "replay_checkpoint_sha256": sha256(args.replay_checkpoint),
            "generalist_actor_sha256": sha256(args.generalist_actor),
            "specialist_actor_sha256": sha256(args.specialist_actor),
        },
        "replay": {
            "frames": len(states),
            "seen_scenarios": len(checkpoint.get("train_seen_scenario_ids", [])),
            "stored_interaction_frames": len(
                replay_state.get("interaction_buffer", [])
            ),
            "derived_interaction_frames": int(np.sum(labels)),
            "temporal_train_frames": split,
            "temporal_validation_frames": len(states) - split,
        },
        "residual_scale": scale.astype(float).tolist(),
        "validation_scale_coverage": coverage.astype(float).tolist(),
        "best_epoch": best[1],
        "train": train_metrics,
        "validation": validation_metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best[2],
            "state_dim": args.state_dim,
            "action_dim": args.action_dim,
            "hidden_dim": args.hidden_dim,
            "residual_scale": scale,
            "best_epoch": best[1],
            "protocol": summary["protocol"],
        },
        args.output_dir / "best_residual_audit.pt",
    )
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
