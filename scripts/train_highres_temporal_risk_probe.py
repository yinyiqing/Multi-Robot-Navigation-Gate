#!/usr/bin/env python3
import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "TD3"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from temporal_risk_models import (  # noqa: E402
    HighResolutionSingleFrameRiskEncoder,
    HighResolutionTemporalRiskEncoder,
)
from temporal_interaction import build_front_lidar_gaps  # noqa: E402
from train_temporal_risk_probe import (  # noqa: E402
    load_first_case_episodes,
    load_json,
    privileged_urgent_label,
    risk_band,
    train_model,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare single-frame and temporal 180-bin lidar risk encoders."
    )
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--development-trajectory", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--test-trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--lidar-dim", type=int, default=180)
    parser.add_argument("--lidar-range", type=float, default=6.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--frame-dim", type=int, default=32)
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


def split_development_cases(scenarios):
    groups = defaultdict(list)
    for scenario in scenarios.values():
        groups[(risk_band(scenario), scenario["preset"])].append(
            scenario["scenario_id"]
        )
    splits = {"train": set(), "validation": set()}
    for group, case_ids in groups.items():
        ordered = sorted(case_ids)
        if len(ordered) != 5:
            raise ValueError(f"Development group {group} must contain five cases")
        splits["train"].update(ordered[:4])
        splits["validation"].add(ordered[4])
    return splits


def raw_points_to_scan(points_by_agent, name, lidar_dim, lidar_range):
    scan = np.full(lidar_dim, lidar_range, dtype=np.float32)
    points = np.asarray(points_by_agent[name], dtype=np.float32)
    if points.size == 0:
        return scan
    bearings = np.arctan2(points[:, 1], points[:, 0])
    gaps = build_front_lidar_gaps(lidar_dim)
    indices = np.searchsorted(gaps[:, 1], bearings, side="right")
    distances = np.linalg.norm(points, axis=1)
    bounded = np.clip(indices, 0, lidar_dim - 1)
    valid = (
        (indices >= 0)
        & (indices < lidar_dim)
        & (bearings >= gaps[bounded, 0])
        & (bearings < gaps[bounded, 1])
    )
    np.minimum.at(scan, indices[valid], np.minimum(distances[valid], lidar_range))
    return scan


def frame_feature(frame, index, name, source, args):
    if source == "raw":
        scan = raw_points_to_scan(
            frame["raw_lidar_points"], name, args.lidar_dim, args.lidar_range
        )
    else:
        scan = np.asarray(frame["temporal_lidar"][name], dtype=np.float32)
        if scan.shape != (args.lidar_dim,):
            raise ValueError(f"Expected {args.lidar_dim} temporal lidar bins")
        scan = np.clip(scan, 0.0, args.lidar_range)
    actor_state = np.asarray(frame["actor_states"][index], dtype=np.float32)
    previous_action = np.clip(actor_state[-2:], -1.0, 1.0)
    return np.concatenate([scan / args.lidar_range, previous_action]).astype(
        np.float32
    )


def append_case_samples(destination, case_id, scenario, frames, source, args):
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
            histories[name].append(frame_feature(frame, index, name, source, args))
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
            destination["x"].append(np.stack(histories[name]))
            destination["y"].append(float(label))
            destination["case"].append(case_id)
        previous_poses = poses


def build_samples(development, development_frames, test, test_frames, args):
    development_splits = split_development_cases(development)
    samples = {
        name: {"x": [], "y": [], "case": []}
        for name in ("train", "validation", "test")
    }
    for case_id, frames in development_frames.items():
        split = next(
            name for name, case_ids in development_splits.items() if case_id in case_ids
        )
        append_case_samples(
            samples[split], case_id, development[case_id], frames, "raw", args
        )
    for case_id, frames in test_frames.items():
        append_case_samples(
            samples["test"], case_id, test[case_id], frames, "temporal", args
        )
    for values in samples.values():
        values["x"] = np.asarray(values["x"], dtype=np.float32)
        values["y"] = np.asarray(values["y"], dtype=np.float32)
    return samples, development_splits


def checked_dataset(manifest_path, trajectory_path):
    payload = load_json(manifest_path)
    scenarios = {item["scenario_id"]: item for item in payload["scenarios"]}
    frames = load_first_case_episodes(trajectory_path)
    if set(frames) != set(scenarios):
        raise ValueError(f"Trajectory cases do not match {manifest_path}")
    return scenarios, frames


def main():
    args = parse_args()
    development, development_frames = checked_dataset(
        args.development_manifest, args.development_trajectory
    )
    test, test_frames = checked_dataset(args.test_manifest, args.test_trajectory)
    if set(development) & set(test):
        raise ValueError("Development and test scenario IDs must be disjoint")
    samples, development_splits = build_samples(
        development, development_frames, test, test_frames, args
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    factories = {
        "mlp": lambda: HighResolutionSingleFrameRiskEncoder(
            args.lidar_dim + 2, args.hidden_dim, args.frame_dim
        ),
        "gru": lambda: HighResolutionTemporalRiskEncoder(
            args.lidar_dim + 2, args.hidden_dim, args.frame_dim
        ),
    }
    results = {}
    for model_name, factory in factories.items():
        model, result = train_model(
            model_name, samples, args, model_factory=factory
        )
        torch.save(model.state_dict(), args.output_dir / f"{model_name}.pth")
        results[model_name] = result

    summary = {
        "protocol": {
            "sequence_length": args.sequence_length,
            "lidar_dim": args.lidar_dim,
            "lidar_range_m": args.lidar_range,
            "input_dim": args.lidar_dim + 2,
            "development_splits": {
                name: sorted(case_ids)
                for name, case_ids in development_splits.items()
            },
            "test_cases": sorted(test),
            "test_locked_for_final_evaluation": True,
            "seed": args.seed,
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
