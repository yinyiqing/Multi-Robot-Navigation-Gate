#!/usr/bin/env python3
import argparse
import gzip
import json
import math
import random
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "TD3"))

from temporal_risk_models import (  # noqa: E402
    SingleFrameRiskEncoder,
    TemporalRiskEncoder,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train deployable single-frame and temporal interaction-risk probes."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--safety-distance", type=float, default=0.9)
    parser.add_argument("--ttc-horizon", type=float, default=4.0)
    parser.add_argument("--max-encounter-distance", type=float, default=4.0)
    parser.add_argument("--target-validation-recall", type=float, default=0.8)
    return parser.parse_args()


def load_json(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def open_text(path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def load_first_case_episodes(path):
    by_episode = defaultdict(list)
    with open_text(path) as handle:
        for line in handle:
            frame = json.loads(line)
            by_episode[int(frame["episode"])].append(frame)
    by_case = {}
    for episode_id in sorted(by_episode):
        frames = by_episode[episode_id]
        by_case.setdefault(frames[0]["case"], frames)
    return by_case


def risk_band(scenario):
    separation = float(scenario["metrics"]["min_synchronized_path_separation_m"])
    if separation < 0.4:
        return "deep"
    if separation < 0.6:
        return "close"
    return "margin"


def split_cases(scenarios):
    groups = defaultdict(list)
    for scenario in scenarios.values():
        groups[(risk_band(scenario), scenario["preset"])].append(
            scenario["scenario_id"]
        )
    splits = {"train": set(), "validation": set(), "test": set()}
    for case_ids in groups.values():
        ordered = sorted(case_ids)
        if len(ordered) != 5:
            raise ValueError("Each risk/pool group must contain exactly five cases")
        splits["train"].update(ordered[:3])
        splits["validation"].add(ordered[3])
        splits["test"].add(ordered[4])
    return splits


def position(pose):
    return np.asarray([pose["x"], pose["y"]], dtype=np.float64)


def wrap_angle(value):
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def privileged_urgent_label(
    name,
    active_names,
    poses,
    previous_poses,
    safety_distance,
    ttc_horizon,
    max_encounter_distance,
):
    if previous_poses is None or name not in previous_poses:
        return False
    ego_pose = poses[name]
    ego_position = position(ego_pose)
    ego_dt = float(ego_pose["timestamp"]) - float(
        previous_poses[name]["timestamp"]
    )
    if ego_dt <= 0.0:
        return False
    ego_velocity = (ego_position - position(previous_poses[name])) / ego_dt
    ego_yaw = float(ego_pose["yaw"])
    cos_yaw, sin_yaw = math.cos(ego_yaw), math.sin(ego_yaw)

    for other_name in active_names:
        if other_name == name or other_name not in previous_poses:
            continue
        other_pose = poses[other_name]
        other_position = position(other_pose)
        other_dt = float(other_pose["timestamp"]) - float(
            previous_poses[other_name]["timestamp"]
        )
        if other_dt <= 0.0:
            continue
        relative_world = other_position - ego_position
        relative_local = np.asarray(
            [
                cos_yaw * relative_world[0] + sin_yaw * relative_world[1],
                -sin_yaw * relative_world[0] + cos_yaw * relative_world[1],
            ]
        )
        distance = float(np.linalg.norm(relative_local))
        bearing = math.atan2(relative_local[1], relative_local[0])
        if (
            distance > max_encounter_distance
            or bearing < -math.pi / 2 - 0.03
            or bearing > math.pi / 2 + 0.03
        ):
            continue
        other_velocity = (
            other_position - position(previous_poses[other_name])
        ) / other_dt
        relative_velocity = other_velocity - ego_velocity
        speed_squared = float(np.dot(relative_velocity, relative_velocity))
        if speed_squared <= 1e-8:
            continue
        closing_speed = -float(
            np.dot(relative_world, relative_velocity)
        ) / max(distance, 1e-6)
        time_to_closest = max(
            -float(np.dot(relative_world, relative_velocity)) / speed_squared,
            0.0,
        )
        closest_distance = float(
            np.linalg.norm(relative_world + relative_velocity * time_to_closest)
        )
        if (
            closing_speed > 0.05
            and time_to_closest <= ttc_horizon
            and closest_distance <= safety_distance
        ):
            return True
    return False


def frame_feature(actor_state):
    state = np.asarray(actor_state, dtype=np.float32)
    if state.shape != (24,):
        raise ValueError("Expected the frozen 5D Actor's 24-dimensional state")
    lidar = np.clip(state[:20], 0.0, 10.0) / 10.0
    previous_action = np.clip(state[-2:], -1.0, 1.0)
    return np.concatenate([lidar, previous_action]).astype(np.float32)


def build_samples(scenarios, frames_by_case, splits, args):
    samples = {name: {"x": [], "y": [], "case": []} for name in splits}
    for case_id, frames in frames_by_case.items():
        split = next(name for name, cases in splits.items() if case_id in cases)
        scenario = scenarios[case_id]
        agent_names = list(scenario["agents"])
        histories = {
            name: deque(maxlen=args.sequence_length) for name in agent_names
        }
        previous_poses = None
        for frame in frames:
            active_names = {
                name
                for index, name in enumerate(agent_names)
                if frame["active_before"][index]
            }
            poses = frame["actor_poses"]
            for index, name in enumerate(agent_names):
                if name not in active_names:
                    continue
                histories[name].append(frame_feature(frame["actor_states"][index]))
                if len(histories[name]) < args.sequence_length:
                    continue
                label = privileged_urgent_label(
                    name,
                    active_names,
                    poses,
                    previous_poses,
                    args.safety_distance,
                    args.ttc_horizon,
                    args.max_encounter_distance,
                )
                samples[split]["x"].append(np.stack(histories[name]))
                samples[split]["y"].append(float(label))
                samples[split]["case"].append(case_id)
            previous_poses = poses
    for split in samples:
        samples[split]["x"] = np.asarray(samples[split]["x"], dtype=np.float32)
        samples[split]["y"] = np.asarray(samples[split]["y"], dtype=np.float32)
    return samples


def binary_metrics(probabilities, targets, threshold):
    predictions = np.asarray(probabilities) >= threshold
    targets = np.asarray(targets, dtype=bool)
    tp = int(np.sum(predictions & targets))
    fp = int(np.sum(predictions & ~targets))
    fn = int(np.sum(~predictions & targets))
    tn = int(np.sum(~predictions & ~targets))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "accuracy": (tp + tn) / (tp + fp + fn + tn),
    }


def choose_threshold(probabilities, targets, target_recall):
    thresholds = np.unique(
        np.concatenate(
            [[0.0, 1.0], np.quantile(probabilities, np.linspace(0, 1, 1001))]
        )
    )
    candidates = []
    for threshold in thresholds:
        result = binary_metrics(probabilities, targets, threshold)
        if result["recall"] >= target_recall:
            candidates.append((threshold, result))
    return min(
        candidates,
        key=lambda item: (
            item[1]["false_positive_rate"],
            -(item[1]["precision"] or 0.0),
            -item[0],
        ),
    )


def predict(model, values, batch_size=512):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            logits = model(torch.from_numpy(values[start : start + batch_size]))
            outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs)


def train_model(model_name, samples, args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model = (
        SingleFrameRiskEncoder(22, args.hidden_dim)
        if model_name == "mlp"
        else TemporalRiskEncoder(22, args.hidden_dim)
    )
    train_x, train_y = samples["train"]["x"], samples["train"]["y"]
    validation_x = samples["validation"]["x"]
    validation_y = samples["validation"]["y"]
    positives = max(float(np.sum(train_y)), 1.0)
    negatives = max(float(len(train_y) - np.sum(train_y)), 1.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives, dtype=torch.float32)
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            loss = criterion(model(batch_x), batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * len(batch_x)
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                criterion(
                    model(torch.from_numpy(validation_x)),
                    torch.from_numpy(validation_y),
                )
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / len(train_x),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    model.load_state_dict(best_state)
    validation_probabilities = predict(model, validation_x)
    threshold, validation_metrics = choose_threshold(
        validation_probabilities,
        validation_y,
        args.target_validation_recall,
    )
    split_metrics = {}
    for split in ("train", "validation", "test"):
        probabilities = predict(model, samples[split]["x"])
        split_metrics[split] = binary_metrics(
            probabilities, samples[split]["y"], threshold
        )
    return model, {
        "model": model_name,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "threshold": float(threshold),
        "threshold_selection": {
            "target_validation_recall": args.target_validation_recall,
            "validation_metrics": validation_metrics,
        },
        "split_metrics": split_metrics,
        "history": history,
    }


def main():
    args = parse_args()
    if args.sequence_length < 2:
        raise ValueError("--sequence-length must be at least 2")
    payload = load_json(args.manifest)
    scenarios = {item["scenario_id"]: item for item in payload["scenarios"]}
    frames_by_case = load_first_case_episodes(args.trajectory)
    if set(frames_by_case) != set(scenarios):
        raise ValueError("Trajectory cases do not exactly cover the manifest")
    splits = split_cases(scenarios)
    samples = build_samples(scenarios, frames_by_case, splits, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for model_name in ("mlp", "gru"):
        model, result = train_model(model_name, samples, args)
        torch.save(model.state_dict(), args.output_dir / f"{model_name}.pth")
        results[model_name] = result
    summary = {
        "protocol": {
            "sequence_length": args.sequence_length,
            "input_dim": 22,
            "input": ["20_bin_front_lidar", "previous_linear_action", "previous_angular_action"],
            "deployable_inputs_only": True,
            "privileged_label": {
                "safety_distance_m": args.safety_distance,
                "ttc_horizon_s": args.ttc_horizon,
                "max_encounter_distance_m": args.max_encounter_distance,
            },
            "seed": args.seed,
            "scenario_splits": {
                name: sorted(cases) for name, cases in splits.items()
            },
        },
        "samples": {
            name: {
                "count": len(values["y"]),
                "positive": int(np.sum(values["y"])),
                "positive_rate": float(np.mean(values["y"])),
            }
            for name, values in samples.items()
        },
        "results": results,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "samples": summary["samples"],
                "mlp": results["mlp"]["split_metrics"],
                "gru": results["gru"]["split_metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
