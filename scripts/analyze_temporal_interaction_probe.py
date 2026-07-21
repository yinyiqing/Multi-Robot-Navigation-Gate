#!/usr/bin/env python3
import argparse
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "TD3"))

from temporal_interaction import TemporalInteractionEncoder, build_front_lidar_gaps


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ego-motion-compensated temporal lidar interaction features."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safety-distance", type=float, default=0.9)
    parser.add_argument("--encounter-distance", type=float, default=1.5)
    parser.add_argument("--urgent-ttc", type=float, default=2.0)
    parser.add_argument("--prediction-persistence", type=int, default=2)
    return parser.parse_args()


def load_json(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_frames(path):
    episodes = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            frame = json.loads(line)
            if "actor_poses" not in frame:
                raise ValueError("Trajectory does not contain actor_poses")
            episodes[int(frame["episode"])].append(frame)
    return episodes


def risk_band(scenario):
    separation = float(scenario["metrics"]["min_synchronized_path_separation_m"])
    if separation < 0.4:
        return "deep"
    if separation < 0.6:
        return "close"
    return "margin"


def empty_confusion():
    return {"tp": 0, "fp": 0, "fn": 0, "tn": 0}


def update_confusion(confusion, prediction, target):
    if prediction and target:
        confusion["tp"] += 1
    elif prediction:
        confusion["fp"] += 1
    elif target:
        confusion["fn"] += 1
    else:
        confusion["tn"] += 1


def confusion_metrics(confusion):
    tp = confusion["tp"]
    fp = confusion["fp"]
    fn = confusion["fn"]
    tn = confusion["tn"]
    return {
        **confusion,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "accuracy": (tp + tn) / (tp + fp + fn + tn),
    }


def ground_truth_urgent(
    name,
    active_names,
    positions,
    previous_positions,
    delta_time,
    safety_distance,
    encounter_distance,
    urgent_ttc,
):
    if previous_positions is None or delta_time <= 0.0:
        return False, None
    minimum_ttc = None
    for other_name in active_names:
        if other_name == name:
            continue
        distance = math.dist(positions[name], positions[other_name])
        previous_distance = math.dist(
            previous_positions[name], previous_positions[other_name]
        )
        closing_speed = (previous_distance - distance) / delta_time
        if distance > encounter_distance or closing_speed <= 0.0:
            continue
        ttc = max(distance - safety_distance, 0.0) / closing_speed
        minimum_ttc = ttc if minimum_ttc is None else min(minimum_ttc, ttc)
    return (
        minimum_ttc is not None and minimum_ttc <= urgent_ttc,
        minimum_ttc,
    )


def evaluate_episode(
    scenario,
    frames,
    safety_distance,
    encounter_distance,
    urgent_ttc,
    persistence,
):
    agent_names = list(scenario["agents"])
    first_temporal_scan = frames[0].get("temporal_lidar")
    scan_dim = (
        len(first_temporal_scan[agent_names[0]])
        if first_temporal_scan is not None
        else 20
    )
    encoders = {
        name: TemporalInteractionEncoder(
            build_front_lidar_gaps(scan_dim),
            collision_distance=0.35,
            ttc_horizon=4.0,
        )
        for name in agent_names
    }
    consecutive_predictions = {name: 0 for name in agent_names}
    frame_confusion = empty_confusion()
    raw_frame_confusion = empty_confusion()
    previous_positions = None
    previous_timestamp = None
    episode_prediction = False
    episode_raw_prediction = False
    episode_target = False
    max_urgency = 0.0
    minimum_predicted_ttc = 4.0
    minimum_ground_truth_ttc = None

    for frame in frames:
        active_names = {
            name
            for index, name in enumerate(agent_names)
            if frame["active_before"][index]
        }
        positions = {
            name: [
                float(frame["actor_poses"][name]["x"]),
                float(frame["actor_poses"][name]["y"]),
            ]
            for name in agent_names
        }
        timestamps = [
            float(frame["actor_poses"][name]["timestamp"])
            for name in active_names
        ]
        current_timestamp = sum(timestamps) / len(timestamps) if timestamps else 0.0
        delta_time = (
            current_timestamp - previous_timestamp
            if previous_timestamp is not None
            else 0.0
        )

        for index, name in enumerate(agent_names):
            if name not in active_names:
                continue
            pose = frame["actor_poses"][name]
            temporal_scan = frame.get("temporal_lidar")
            scan = (
                temporal_scan[name]
                if temporal_scan is not None
                else frame["actor_states"][index][:20]
            )
            result = encoders[name].update(
                scan,
                [pose["x"], pose["y"], pose["yaw"]],
                pose["timestamp"],
            )
            urgency = float(result["summary"][0])
            predicted_ttc = float(result["summary"][1]) * 4.0
            raw_prediction = predicted_ttc <= urgent_ttc
            consecutive_predictions[name] = (
                consecutive_predictions[name] + 1 if raw_prediction else 0
            )
            prediction = consecutive_predictions[name] >= persistence
            target, target_ttc = ground_truth_urgent(
                name,
                active_names,
                positions,
                previous_positions,
                delta_time,
                safety_distance,
                encounter_distance,
                urgent_ttc,
            )
            update_confusion(raw_frame_confusion, raw_prediction, target)
            update_confusion(frame_confusion, prediction, target)
            episode_raw_prediction |= raw_prediction
            episode_prediction |= prediction
            episode_target |= target
            max_urgency = max(max_urgency, urgency)
            minimum_predicted_ttc = min(minimum_predicted_ttc, predicted_ttc)
            if target_ttc is not None:
                minimum_ground_truth_ttc = (
                    target_ttc
                    if minimum_ground_truth_ttc is None
                    else min(minimum_ground_truth_ttc, target_ttc)
                )

        previous_positions = positions
        previous_timestamp = current_timestamp

    return {
        "case": scenario["scenario_id"],
        "risk_band": risk_band(scenario),
        "pool": scenario["preset"],
        "episode_target": episode_target,
        "episode_prediction": episode_prediction,
        "episode_raw_prediction": episode_raw_prediction,
        "max_urgency": max_urgency,
        "minimum_predicted_ttc_s": minimum_predicted_ttc,
        "minimum_ground_truth_ttc_s": minimum_ground_truth_ttc,
        "lidar_sectors": scan_dim,
        "frame_confusion": frame_confusion,
        "raw_frame_confusion": raw_frame_confusion,
    }


def combine_confusions(items, key):
    combined = empty_confusion()
    for item in items:
        for field in combined:
            combined[field] += item[key][field]
    return confusion_metrics(combined)


def summarize_episodes(items):
    confusion = empty_confusion()
    for item in items:
        update_confusion(
            confusion, item["episode_prediction"], item["episode_target"]
        )
    episodes = len(items)
    return {
        "episodes": episodes,
        "predicted_activation_rate": sum(
            item["episode_prediction"] for item in items
        )
        / episodes,
        "ground_truth_urgent_rate": sum(item["episode_target"] for item in items)
        / episodes,
        "confusion": confusion_metrics(confusion),
    }


def main():
    args = parse_args()
    if args.safety_distance <= 0.0:
        raise ValueError("--safety-distance must be positive")
    if args.encounter_distance <= args.safety_distance:
        raise ValueError("--encounter-distance must exceed --safety-distance")
    if args.urgent_ttc <= 0.0:
        raise ValueError("--urgent-ttc must be positive")
    if args.prediction_persistence < 1:
        raise ValueError("--prediction-persistence must be positive")

    payload = load_json(args.manifest)
    scenarios = {item["scenario_id"]: item for item in payload["scenarios"]}
    frames = load_frames(args.trajectory)
    frame_cases = {episode_frames[0]["case"] for episode_frames in frames.values()}
    if frame_cases != set(scenarios):
        raise ValueError("Trajectory cases do not exactly cover the manifest")

    episodes = []
    for episode_frames in frames.values():
        scenario = scenarios[episode_frames[0]["case"]]
        episodes.append(
            evaluate_episode(
                scenario,
                episode_frames,
                args.safety_distance,
                args.encounter_distance,
                args.urgent_ttc,
                args.prediction_persistence,
            )
        )

    result = {
        "protocol": {
            "safety_distance_m": args.safety_distance,
            "encounter_distance_m": args.encounter_distance,
            "urgent_ttc_s": args.urgent_ttc,
            "prediction_persistence_frames": args.prediction_persistence,
            "deployable_inputs": ["front_lidar", "self_odometry", "timestamp"],
            "ground_truth_only_inputs": ["other_robot_positions"],
            "lidar_sectors": sorted({item["lidar_sectors"] for item in episodes}),
        },
        "frame_metrics": combine_confusions(episodes, "frame_confusion"),
        "raw_frame_metrics": combine_confusions(episodes, "raw_frame_confusion"),
        "episode_metrics": summarize_episodes(episodes),
        "by_risk_band": {
            band: summarize_episodes(
                [item for item in episodes if item["risk_band"] == band]
            )
            for band in ("deep", "close", "margin")
        },
        "episodes": sorted(episodes, key=lambda item: item["case"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("frame_metrics", "episode_metrics", "by_risk_band")
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
