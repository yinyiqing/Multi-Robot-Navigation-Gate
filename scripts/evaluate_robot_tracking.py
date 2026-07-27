#!/usr/bin/env python3
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "TD3"))

from robot_perception.dataset import list_shards, load_shard
from robot_perception.metrics import detection_metrics, select_validation_threshold
from robot_perception.models import LocalRobotDetector
from robot_perception.tracker import RobotCandidateTracker


TRACKING_KEYS = {
    "ego_indices",
    "ego_poses",
    "frame_indices",
    "target_agent_indices",
    "timestamps",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate deployable candidate tracking on v2 perception shards."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--association-distance", type=float, default=0.65)
    parser.add_argument("--probability-alpha", type=float, default=0.35)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


@torch.no_grad()
def predict(model, patches, batch_size, device):
    probabilities = []
    for start in range(0, len(patches), batch_size):
        values = torch.from_numpy(
            patches[start : start + batch_size].astype(np.float32)
        ).to(device)
        probabilities.append(torch.sigmoid(model(values)[0]).cpu().numpy())
    return np.concatenate(probabilities) if probabilities else np.empty(0)


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"count": 0, "p10": None, "median": None, "p90": None}
    return {
        "count": len(values),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
    }


def evaluate_tracking(model, paths, args):
    raw_probabilities = []
    smoothed_probabilities = []
    labels = []
    ages = []
    dynamic_speeds = []
    visible_count = 0
    track_identities = defaultdict(list)
    identity_tracks = defaultdict(set)

    for path in paths:
        shard = load_shard(path)
        missing = TRACKING_KEYS - set(shard)
        if missing:
            raise ValueError(f"v2 tracking shard {path} is missing {sorted(missing)}")
        probabilities = predict(model, shard["patches"], args.batch_size, args.device)
        smoothed = np.zeros_like(probabilities)
        candidate_ages = np.zeros(len(probabilities), dtype=np.int32)
        candidate_speeds = np.zeros(len(probabilities), dtype=np.float32)
        scenario_id = str(shard["scenario_id"])

        for ego_index in np.unique(shard["ego_indices"]):
            tracker = RobotCandidateTracker(
                association_distance=args.association_distance,
                probability_alpha=args.probability_alpha,
            )
            ego_rows = np.flatnonzero(shard["ego_indices"] == ego_index)
            for frame_index in sorted(np.unique(shard["frame_indices"][ego_rows])):
                rows = ego_rows[shard["frame_indices"][ego_rows] == frame_index]
                tracked = tracker.update(
                    shard["candidate_centers"][rows],
                    probabilities[rows],
                    shard["ego_poses"][rows[0]],
                    shard["timestamps"][rows[0]],
                )
                for item in tracked:
                    row = rows[item.observation_index]
                    smoothed[row] = item.smoothed_shape_probability
                    candidate_ages[row] = item.age
                    candidate_speeds[row] = item.dynamic_speed
                    target_id = int(shard["target_agent_indices"][row])
                    track_key = (scenario_id, int(ego_index), item.track_id)
                    track_identities[track_key].append(target_id)
                    if target_id >= 0:
                        identity_tracks[(scenario_id, int(ego_index), target_id)].add(
                            item.track_id
                        )

        raw_probabilities.append(probabilities)
        smoothed_probabilities.append(smoothed)
        labels.append(shard["labels"].astype(np.uint8))
        ages.append(candidate_ages)
        dynamic_speeds.append(candidate_speeds)
        visible_count += int(shard["visible_robot_count"])

    raw_probabilities = np.concatenate(raw_probabilities)
    smoothed_probabilities = np.concatenate(smoothed_probabilities)
    labels = np.concatenate(labels)
    ages = np.concatenate(ages)
    dynamic_speeds = np.concatenate(dynamic_speeds)
    tracked_mask = ages >= 2
    positive_tracks = [
        values for values in track_identities.values() if any(value >= 0 for value in values)
    ]
    mixed_tracks = [
        values
        for values in positive_tracks
        if any(value < 0 for value in values)
    ]
    identity_switches = [
        values
        for values in positive_tracks
        if len({value for value in values if value >= 0}) > 1
    ]
    return {
        "raw_probabilities": raw_probabilities,
        "smoothed_probabilities": smoothed_probabilities,
        "labels": labels,
        "visible_count": visible_count,
        "motion": {
            "minimum_track_age": 2,
            "positive_dynamic_speed_mps": quantiles(
                dynamic_speeds[tracked_mask & (labels == 1)]
            ),
            "negative_dynamic_speed_mps": quantiles(
                dynamic_speeds[tracked_mask & (labels == 0)]
            ),
        },
        "association": {
            "tracks_with_robot_labels": len(positive_tracks),
            "tracks_mixing_positive_and_negative_labels": len(mixed_tracks),
            "tracks_mixing_multiple_robot_ids": len(identity_switches),
            "robot_identities_observed": len(identity_tracks),
            "mean_track_fragments_per_robot_identity": (
                float(np.mean([len(value) for value in identity_tracks.values()]))
                if identity_tracks
                else None
            ),
        },
    }


def main():
    args = parse_args()
    checkpoint = torch.load(
        args.checkpoint, map_location=args.device, weights_only=False
    )
    model = LocalRobotDetector(**checkpoint.get("model_config", {})).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    values = evaluate_tracking(model, list_shards(args.shard_dir), args)
    raw_metrics = detection_metrics(
        values["raw_probabilities"],
        values["labels"],
        values["visible_count"],
        threshold=float(checkpoint["threshold"]),
    )
    smoothed_metrics = select_validation_threshold(
        values["smoothed_probabilities"],
        values["labels"],
        values["visible_count"],
    )
    result = {
        "shards": len(list_shards(args.shard_dir)),
        "candidates": len(values["labels"]),
        "raw_g0": raw_metrics.to_dict(),
        "smoothed_g0": smoothed_metrics.to_dict(),
        "motion": values["motion"],
        "association": values["association"],
        "config": {
            "association_distance": args.association_distance,
            "probability_alpha": args.probability_alpha,
        },
    }
    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
