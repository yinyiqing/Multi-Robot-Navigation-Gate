#!/usr/bin/env python3
import argparse
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
from policy_consolidation import (
    actor_actions,
    consolidation_metrics,
    initialize_augmented_actor,
    load_augmented_consolidation_dataset,
    load_consolidation_dataset,
    scenario_class_weights,
    select_teacher_actions,
)
from robot_perception.dataset import list_shards
from robot_perception.models import LocalRobotDetector


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distill the privileged 5A/local-specialist composition into Actor B."
    )
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--generalist-actor", type=Path, required=True)
    parser.add_argument("--specialist-actor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--input-mode",
        choices=("actor-state", "deployable-gate"),
        default="actor-state",
    )
    parser.add_argument("--detector-checkpoint", type=Path)
    parser.add_argument("--max-tracks", type=int, default=4)
    parser.add_argument("--label", choices=("any", "front"), default="any")
    parser.add_argument(
        "--initialization", choices=("specialist", "generalist"), default="specialist"
    )
    parser.add_argument("--state-dim", type=int, default=24)
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--disagreement-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--device", default="cpu")
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


def model_metrics(model, dataset, teachers, args):
    actions = actor_actions(
        model, dataset["features"], args.batch_size, args.device
    )
    return consolidation_metrics(
        actions,
        teachers["generalist"],
        teachers["specialist"],
        dataset["labels"],
        strata=dataset["strata"],
        disagreement_threshold=args.disagreement_threshold,
    )


def selection_key(metrics):
    return (
        max(metrics["normal"]["mse"], metrics["interaction"]["mse"]),
        metrics["all"]["mse"],
    )


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    if args.state_dim < 1 or args.action_dim < 1:
        raise ValueError("actor dimensions must be positive")
    if args.learning_rate <= 0.0 or args.disagreement_threshold < 0.0:
        raise ValueError("learning rate must be positive and threshold non-negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_paths = list_shards(args.train_dir)
    validation_paths = list_shards(args.validation_dir)
    if args.input_mode == "deployable-gate":
        if args.detector_checkpoint is None:
            raise ValueError("deployable-gate input requires --detector-checkpoint")
        detector_checkpoint = torch.load(
            args.detector_checkpoint,
            map_location=args.device,
            weights_only=False,
        )
        detector = LocalRobotDetector(
            **detector_checkpoint.get("model_config", {})
        ).to(args.device)
        detector.load_state_dict(detector_checkpoint["model_state_dict"])
        detector.eval()
        load_dataset = lambda paths: load_augmented_consolidation_dataset(
            paths,
            detector,
            args.batch_size,
            args.device,
            label=args.label,
            state_dim=args.state_dim,
            max_tracks=args.max_tracks,
        )
    else:
        load_dataset = lambda paths: load_consolidation_dataset(
            paths, label=args.label, state_dim=args.state_dim
        )
    train = load_dataset(train_paths)
    validation = load_dataset(validation_paths)
    train_ids = set(train["scenarios"].tolist())
    validation_ids = set(validation["scenarios"].tolist())
    overlap = train_ids & validation_ids
    if overlap:
        raise ValueError(
            f"train and validation share {len(overlap)} scenario IDs"
        )

    generalist = load_actor(
        args.generalist_actor, args.state_dim, args.action_dim, args.device
    )
    specialist = load_actor(
        args.specialist_actor, args.state_dim, args.action_dim, args.device
    )
    teachers = {}
    validation_teachers = {}
    for name, actor in (("generalist", generalist), ("specialist", specialist)):
        teachers[name] = actor_actions(
            actor, train["states"], args.batch_size, args.device
        )
        validation_teachers[name] = actor_actions(
            actor, validation["states"], args.batch_size, args.device
        )
    targets = select_teacher_actions(
        teachers["generalist"], teachers["specialist"], train["labels"]
    )
    weights = scenario_class_weights(train["labels"], train["scenarios"])

    student_input_dim = train["features"].shape[1]
    if validation["features"].shape[1] != student_input_dim:
        raise ValueError("train and validation student feature dimensions differ")
    student = Actor(student_input_dim, args.action_dim).to(args.device)
    initialization_actor = specialist if args.initialization == "specialist" else generalist
    initialize_augmented_actor(student, initialization_actor, args.state_dim)
    optimizer = torch.optim.Adam(
        student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    initial = model_metrics(student, validation, validation_teachers, args)
    history = []
    best = None
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        student.train()
        losses = []
        for start in range(0, len(train["states"]), args.batch_size):
            if start == 0:
                indices = rng.permutation(len(train["states"]))
            batch = indices[start : start + args.batch_size]
            states = torch.from_numpy(train["features"][batch]).to(args.device)
            target = torch.from_numpy(targets[batch]).to(args.device)
            batch_weights = torch.from_numpy(weights[batch]).to(args.device)
            per_frame = F.mse_loss(student(states), target, reduction="none").mean(1)
            loss = torch.mean(per_frame * batch_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        student.eval()
        metrics = model_metrics(student, validation, validation_teachers, args)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation": metrics,
        }
        history.append(record)
        key = selection_key(metrics)
        if best is None or key < best[0]:
            best = (key, epoch, metrics)
            torch.save(student.state_dict(), args.output_dir / "best_actor.pth")
            torch.save(
                {
                    "format_version": 1,
                    "epoch": epoch,
                    "actor_state_dict": student.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_metrics": metrics,
                    "config": vars(args),
                },
                args.output_dir / "best.pt",
            )
        print(
            "epoch=%d loss=%.6f val_mse=%.6f normal_mse=%.6f "
            "interaction_mse=%.6f choice=%.3f"
            % (
                epoch,
                record["train_loss"],
                metrics["all"]["mse"],
                metrics["normal"]["mse"],
                metrics["interaction"]["mse"],
                metrics["teacher_choice_accuracy"] or 0.0,
            )
        )

    best_actor = load_actor(
        args.output_dir / "best_actor.pth",
        student_input_dim,
        args.action_dim,
        args.device,
    )
    best_metrics = model_metrics(
        best_actor, validation, validation_teachers, args
    )
    summary = {
        "protocol": "privileged-oracle-dual-teacher-consolidation-pilot-v1",
        "initialization": args.initialization,
        "input_mode": args.input_mode,
        "student_input_dim": student_input_dim,
        "label": args.label,
        "train": {
            "scenarios": len(train_ids),
            "frames": len(train["labels"]),
            "interaction_rate": float(np.mean(train["labels"])),
        },
        "validation": {
            "scenarios": len(validation_ids),
            "frames": len(validation["labels"]),
            "interaction_rate": float(np.mean(validation["labels"])),
        },
        "actors": {
            "generalist": {
                "path": str(args.generalist_actor),
                "sha256": sha256(args.generalist_actor),
            },
            "specialist": {
                "path": str(args.specialist_actor),
                "sha256": sha256(args.specialist_actor),
            },
            "detector": (
                {
                    "path": str(args.detector_checkpoint),
                    "sha256": sha256(args.detector_checkpoint),
                }
                if args.detector_checkpoint is not None
                else None
            ),
        },
        "initial_validation": initial,
        "best_epoch": best[1],
        "best_validation": best_metrics,
        "selection": "minimize max(normal_mse, interaction_mse), then total_mse",
        "limitations": [
            "Pilot states were collected under the frozen 5A policy, not the Oracle-composed teacher.",
            "Offline action fidelity does not establish closed-loop navigation performance.",
        ],
    }
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
