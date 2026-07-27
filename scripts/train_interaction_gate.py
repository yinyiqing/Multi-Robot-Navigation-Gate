#!/usr/bin/env python3
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "TD3"))

from interaction_gate import InteractionGate
from robot_perception.dataset import list_shards, load_shard
from robot_perception.gate_features import build_gate_feature, gate_feature_dim
from robot_perception.models import LocalRobotDetector
from robot_perception.tracker import RobotCandidateTracker


GATE_KEYS = {
    "frame_actor_states",
    "frame_ego_indices",
    "frame_ego_poses",
    "frame_front_interaction_labels",
    "frame_indices_unique",
    "frame_oracle_interaction_labels",
    "frame_record_indices",
    "frame_timestamps",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train the G2-A Oracle imitation Gate.")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", choices=("any", "front"), required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-tracks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


@torch.no_grad()
def detector_probabilities(model, patches, batch_size, device):
    probabilities = []
    for start in range(0, len(patches), batch_size):
        values = torch.from_numpy(
            patches[start : start + batch_size].astype(np.float32)
        ).to(device)
        probabilities.append(torch.sigmoid(model(values)[0]).cpu().numpy())
    return np.concatenate(probabilities) if probabilities else np.empty(0)


def build_dataset(paths, detector, args):
    all_features = []
    all_labels = []
    all_scenarios = []
    all_strata = []
    label_key = (
        "frame_oracle_interaction_labels"
        if args.label == "any"
        else "frame_front_interaction_labels"
    )
    for path in paths:
        shard = load_shard(path)
        missing = GATE_KEYS - set(shard)
        if missing:
            raise ValueError(f"gate shard {path} is missing {sorted(missing)}")
        probabilities = detector_probabilities(
            detector, shard["patches"], args.batch_size, args.device
        )
        frame_candidates = [[] for _ in range(len(shard["frame_ego_indices"]))]
        for candidate_index, frame_index in enumerate(shard["frame_record_indices"]):
            frame_candidates[int(frame_index)].append(candidate_index)
        shard_features = np.zeros(
            (len(frame_candidates), gate_feature_dim(24, args.max_tracks)),
            dtype=np.float32,
        )
        for ego_index in np.unique(shard["frame_ego_indices"]):
            tracker = RobotCandidateTracker()
            frame_rows = np.flatnonzero(shard["frame_ego_indices"] == ego_index)
            frame_rows = frame_rows[
                np.argsort(shard["frame_indices_unique"][frame_rows], kind="stable")
            ]
            for frame_row in frame_rows:
                candidate_rows = np.asarray(
                    frame_candidates[int(frame_row)], dtype=np.int64
                )
                tracked = tracker.update(
                    shard["candidate_centers"][candidate_rows],
                    probabilities[candidate_rows],
                    shard["frame_ego_poses"][frame_row],
                    shard["frame_timestamps"][frame_row],
                )
                shard_features[frame_row] = build_gate_feature(
                    shard["frame_actor_states"][frame_row],
                    tracked,
                    max_tracks=args.max_tracks,
                )
        count = len(shard_features)
        all_features.append(shard_features)
        all_labels.append(shard[label_key].astype(np.float32))
        all_scenarios.extend([str(shard["scenario_id"])] * count)
        all_strata.extend(
            [f"{str(shard['scenario_pool'])}_{str(shard['interaction_band'])}"]
            * count
        )
    return {
        "features": np.concatenate(all_features),
        "labels": np.concatenate(all_labels),
        "scenarios": np.asarray(all_scenarios),
        "strata": np.asarray(all_strata),
    }


def binary_metrics(probabilities, labels, threshold):
    labels = np.asarray(labels, dtype=bool)
    predictions = np.asarray(probabilities) >= threshold
    true_positive = int(np.sum(predictions & labels))
    false_positive = int(np.sum(predictions & ~labels))
    false_negative = int(np.sum(~predictions & labels))
    true_negative = int(np.sum(~predictions & ~labels))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    fpr = false_positive / max(false_positive + true_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "f1": float(f1),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "meets_entry_criteria": bool(recall >= 0.90 and fpr <= 0.10),
    }


def gate_metrics(probabilities, labels, threshold, guard_mask=None):
    metrics = binary_metrics(probabilities, labels, threshold)
    if guard_mask is None:
        return metrics
    guard_mask = np.asarray(guard_mask, dtype=bool)
    if guard_mask.shape != np.asarray(labels).shape or not np.any(guard_mask):
        raise ValueError("guard_mask must select at least one validation frame")
    guard = binary_metrics(
        np.asarray(probabilities)[guard_mask],
        np.asarray(labels)[guard_mask],
        threshold,
    )
    metrics["standard_weak_fpr"] = guard["fpr"]
    metrics["meets_entry_criteria"] = bool(
        metrics["meets_entry_criteria"] and guard["fpr"] <= 0.10
    )
    return metrics


def select_threshold(probabilities, labels, guard_mask=None):
    metrics = [
        gate_metrics(probabilities, labels, threshold, guard_mask)
        for threshold in np.linspace(0.01, 0.99, 99)
    ]
    feasible = [item for item in metrics if item["meets_entry_criteria"]]
    if feasible:
        return max(feasible, key=lambda item: (item["precision"], item["f1"]))
    return max(metrics, key=lambda item: (item["f1"], item["recall"], -item["fpr"]))


def sample_weights(labels, scenarios):
    scenario_counts = Counter(scenarios.tolist())
    weights = np.asarray([1.0 / scenario_counts[item] for item in scenarios])
    class_totals = [float(np.sum(weights[labels == value])) for value in (0, 1)]
    for value in (0, 1):
        if class_totals[value] > 0.0:
            weights[labels == value] *= 0.5 / class_totals[value]
    return (weights / np.mean(weights)).astype(np.float32)


@torch.no_grad()
def predict_gate(model, features, batch_size, device):
    probabilities = []
    for start in range(0, len(features), batch_size):
        values = torch.from_numpy(features[start : start + batch_size]).to(device)
        probabilities.append(torch.sigmoid(model(values)).cpu().numpy())
    return np.concatenate(probabilities)


def lidar_rule(validation, actor_state_dim=24, guard_mask=None):
    minimum_lidar = np.min(validation["features"][:, : actor_state_dim - 4], axis=1)
    candidates = []
    for distance in np.linspace(0.2, 4.0, 77):
        probabilities = (minimum_lidar <= distance).astype(np.float32)
        candidates.append(
            gate_metrics(probabilities, validation["labels"], 0.5, guard_mask)
        )
        candidates[-1]["distance_m"] = float(distance)
    feasible = [item for item in candidates if item["meets_entry_criteria"]]
    return max(
        feasible or candidates,
        key=lambda item: (item["f1"], item["recall"], -item["fpr"]),
    )


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_tracks < 1:
        raise ValueError("training dimensions must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    detector_checkpoint = torch.load(
        args.detector_checkpoint, map_location=args.device, weights_only=False
    )
    detector = LocalRobotDetector(
        **detector_checkpoint.get("model_config", {})
    ).to(args.device)
    detector.load_state_dict(detector_checkpoint["model_state_dict"])
    detector.eval()
    train = build_dataset(list_shards(args.train_dir), detector, args)
    validation = build_dataset(list_shards(args.validation_dir), detector, args)
    mean = np.mean(train["features"], axis=0).astype(np.float32)
    std = np.std(train["features"], axis=0).astype(np.float32)
    std[std < 1e-3] = 1.0
    train_features = (train["features"] - mean) / std
    validation_features = (validation["features"] - mean) / std
    standard_weak_mask = validation["strata"] == "standard_weak"
    weights = sample_weights(train["labels"], train["scenarios"])
    model = InteractionGate(train_features.shape[1]).to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng = np.random.default_rng(args.seed)
    history = []
    best = None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        indices = rng.permutation(len(train_features))
        model.train()
        losses = []
        for start in range(0, len(indices), args.batch_size):
            batch = indices[start : start + args.batch_size]
            features = torch.from_numpy(train_features[batch]).to(args.device)
            labels = torch.from_numpy(train["labels"][batch]).to(args.device)
            batch_weights = torch.from_numpy(weights[batch]).to(args.device)
            loss = (
                F.binary_cross_entropy_with_logits(
                    model(features), labels, reduction="none"
                )
                * batch_weights
            ).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        probabilities = predict_gate(
            model, validation_features, args.batch_size, args.device
        )
        metrics = select_threshold(
            probabilities, validation["labels"], standard_weak_mask
        )
        record = {"epoch": epoch, "loss": float(np.mean(losses)), **metrics}
        history.append(record)
        key = (
            int(metrics["meets_entry_criteria"]),
            metrics["f1"],
            metrics["recall"],
            -metrics["fpr"],
        )
        if best is None or key > best[0]:
            best = (key, record)
            torch.save(
                {
                    "format_version": 1,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "input_dim": train_features.shape[1],
                        "hidden_dims": (128, 64),
                    },
                    "feature_mean": mean,
                    "feature_std": std,
                    "max_tracks": args.max_tracks,
                    "label": args.label,
                    "threshold": metrics["threshold"],
                    "validation_metrics": metrics,
                },
                args.output_dir / "best.pt",
            )
        print(
            "epoch=%d loss=%.5f threshold=%.2f precision=%.3f recall=%.3f "
            "fpr=%.3f f1=%.3f pass=%s"
            % (
                epoch,
                record["loss"],
                metrics["threshold"],
                metrics["precision"],
                metrics["recall"],
                metrics["fpr"],
                metrics["f1"],
                metrics["meets_entry_criteria"],
            )
        )
    checkpoint = torch.load(
        args.output_dir / "best.pt", map_location=args.device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    probabilities = predict_gate(model, validation_features, args.batch_size, args.device)
    metrics = gate_metrics(
        probabilities,
        validation["labels"],
        checkpoint["threshold"],
        standard_weak_mask,
    )
    strata = {}
    for name in sorted(set(validation["strata"].tolist())):
        mask = validation["strata"] == name
        strata[name] = binary_metrics(
            probabilities[mask], validation["labels"][mask], checkpoint["threshold"]
        )
    summary = {
        "label": args.label,
        "train_frames": len(train["labels"]),
        "validation_frames": len(validation["labels"]),
        "train_positive_rate": float(np.mean(train["labels"])),
        "validation_positive_rate": float(np.mean(validation["labels"])),
        "best_epoch": checkpoint["epoch"],
        "validation": metrics,
        "validation_strata": strata,
        "minimum_lidar_rule": lidar_rule(validation, guard_mask=standard_weak_mask),
    }
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
