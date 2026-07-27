#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "TD3"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from robot_perception.dataset import list_shards, load_shard
from robot_perception.metrics import detection_metrics
from robot_perception.models import LocalRobotDetector
from train_robot_detector import evaluate


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen robot detector at its validation-selected threshold."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = torch.load(
        args.checkpoint, map_location=args.device, weights_only=False
    )
    model = LocalRobotDetector(**checkpoint.get("model_config", {})).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    shard_paths = list_shards(args.shard_dir)
    probabilities, labels, visible_count, offset_mae = evaluate(
        model, shard_paths, args.batch_size, args.device
    )
    metrics = detection_metrics(
        probabilities,
        labels,
        visible_count,
        threshold=float(checkpoint["threshold"]),
    )
    result = metrics.to_dict()
    result["center_offset_mae_m"] = offset_mae
    grouped_paths = {}
    for path in shard_paths:
        shard = load_shard(path)
        pool = str(shard.get("scenario_pool", np.asarray("unknown")))
        band = str(shard.get("interaction_band", np.asarray("unknown")))
        grouped_paths.setdefault(f"{pool}_{band}", []).append(path)
    result["strata"] = {}
    for name, paths in sorted(grouped_paths.items()):
        group_probabilities, group_labels, group_visible, group_offset_mae = evaluate(
            model, paths, args.batch_size, args.device
        )
        group_metrics = detection_metrics(
            group_probabilities,
            group_labels,
            group_visible,
            threshold=float(checkpoint["threshold"]),
        ).to_dict()
        group_metrics["center_offset_mae_m"] = group_offset_mae
        result["strata"][name] = group_metrics
    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
