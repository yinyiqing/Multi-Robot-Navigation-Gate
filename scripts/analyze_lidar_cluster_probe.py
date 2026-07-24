#!/usr/bin/env python3
import argparse
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "TD3"))

from lidar_cluster_tracking import LidarClusterTracker, world_to_local  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate deployable lidar cluster CPA/TTC observations."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--truth-safety-distance", type=float, default=0.9)
    parser.add_argument("--predicted-collision-distance", type=float, default=0.75)
    parser.add_argument("--ttc-horizon", type=float, default=4.0)
    parser.add_argument("--max-encounter-distance", type=float, default=4.0)
    parser.add_argument("--min-track-age", type=int, default=3)
    parser.add_argument("--association-distance", type=float, default=0.45)
    parser.add_argument("--dynamic-speed-deadband", type=float, default=0.2)
    return parser.parse_args()


def load_json(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def load_frames(path):
    episodes = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            frame = json.loads(line)
            if frame.get("raw_lidar_points") is None:
                raise ValueError("Trajectory does not contain raw_lidar_points")
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
    tp, fp = confusion["tp"], confusion["fp"]
    fn, tn = confusion["fn"], confusion["tn"]
    return {
        **confusion,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "accuracy": (tp + tn) / (tp + fp + fn + tn),
    }


def pose_position(pose):
    return np.array([float(pose["x"]), float(pose["y"])], dtype=np.float64)


def ground_truth_urgent(
    name,
    active_names,
    poses,
    previous_poses,
    safety_distance,
    ttc_horizon,
    max_encounter_distance,
):
    if previous_poses is None or name not in previous_poses:
        return False, None, []
    ego_pose = poses[name]
    ego_position = pose_position(ego_pose)
    ego_dt = float(ego_pose["timestamp"]) - float(
        previous_poses[name]["timestamp"]
    )
    if ego_dt <= 0.0:
        return False, None, []
    ego_velocity = (
        ego_position - pose_position(previous_poses[name])
    ) / ego_dt
    minimum_ttc = None
    urgent_relative_positions = []

    for other_name in active_names:
        if other_name == name or other_name not in previous_poses:
            continue
        other_pose = poses[other_name]
        other_position = pose_position(other_pose)
        other_dt = float(other_pose["timestamp"]) - float(
            previous_poses[other_name]["timestamp"]
        )
        if other_dt <= 0.0:
            continue
        relative_local = world_to_local(
            other_position,
            [ego_position[0], ego_position[1], float(ego_pose["yaw"])],
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
            other_position - pose_position(previous_poses[other_name])
        ) / other_dt
        relative_position = other_position - ego_position
        relative_velocity = other_velocity - ego_velocity
        speed_squared = float(np.dot(relative_velocity, relative_velocity))
        if speed_squared <= 1e-8:
            continue
        closing_speed = -float(
            np.dot(relative_position, relative_velocity)
        ) / max(distance, 1e-6)
        time_to_closest = max(
            -float(np.dot(relative_position, relative_velocity)) / speed_squared,
            0.0,
        )
        closest_distance = float(
            np.linalg.norm(relative_position + relative_velocity * time_to_closest)
        )
        if (
            closing_speed > 0.05
            and time_to_closest <= ttc_horizon
            and closest_distance <= safety_distance
        ):
            urgent_relative_positions.append(relative_local)
            minimum_ttc = (
                time_to_closest
                if minimum_ttc is None
                else min(minimum_ttc, time_to_closest)
            )
    return minimum_ttc is not None, minimum_ttc, urgent_relative_positions


def combine_confusions(items, key):
    combined = empty_confusion()
    for item in items:
        for field in combined:
            combined[field] += item[key][field]
    return confusion_metrics(combined)


def summarize_episodes(items):
    confusion = empty_confusion()
    for item in items:
        update_confusion(confusion, item["prediction"], item["target"])
    return {
        "episodes": len(items),
        "confusion": confusion_metrics(confusion),
        "predicted_activation_rate": sum(item["prediction"] for item in items)
        / len(items),
        "ground_truth_urgent_rate": sum(item["target"] for item in items)
        / len(items),
    }


def evaluate_episode(scenario, frames, args):
    agent_names = list(scenario["agents"])
    trackers = {
        name: LidarClusterTracker(
            association_distance=args.association_distance,
            collision_distance=args.predicted_collision_distance,
            ttc_horizon=args.ttc_horizon,
            dynamic_speed_deadband=args.dynamic_speed_deadband,
        )
        for name in agent_names
    }
    frame_confusion = empty_confusion()
    previous_poses = None
    episode_prediction = False
    episode_target = False
    tracked_frames = 0
    detected_clusters = 0
    mature_tracks = 0
    minimum_predicted_ttc = None
    minimum_truth_ttc = None
    truth_agent_frames = 0
    truth_with_raw_support = 0
    truth_with_cluster_support = 0
    truth_with_mature_track_support = 0
    false_positive_agent_frames = 0
    false_positive_with_robot_support = 0

    for frame in frames:
        poses = frame["actor_poses"]
        active_names = {
            name
            for index, name in enumerate(agent_names)
            if frame["active_before"][index]
        }
        for name in active_names:
            pose = poses[name]
            tracks = trackers[name].update(
                frame["raw_lidar_points"][name],
                [pose["x"], pose["y"], pose["yaw"]],
                pose["timestamp"],
            )
            detected_clusters += len(tracks)
            mature_tracks += sum(track["age"] >= args.min_track_age for track in tracks)
            prediction_tracks = [
                track
                for track in tracks
                if track["age"] >= args.min_track_age and track["urgent"]
            ]
            prediction = bool(prediction_tracks)
            target, truth_ttc, truth_positions = ground_truth_urgent(
                name,
                active_names,
                poses,
                previous_poses,
                args.truth_safety_distance,
                args.ttc_horizon,
                args.max_encounter_distance,
            )
            if target:
                truth_agent_frames += 1
                raw_points = np.asarray(
                    frame["raw_lidar_points"][name], dtype=np.float64
                )[:, :2]
                cluster_centroids = [track["centroid"] for track in tracks]
                mature_centroids = [
                    track["centroid"]
                    for track in tracks
                    if track["age"] >= args.min_track_age
                ]

                def has_support(points, threshold):
                    if len(points) == 0:
                        return False
                    values = np.asarray(points, dtype=np.float64)
                    return any(
                        float(np.min(np.linalg.norm(values - target_position, axis=1)))
                        <= threshold
                        for target_position in truth_positions
                    )

                truth_with_raw_support += int(has_support(raw_points, 0.45))
                truth_with_cluster_support += int(
                    has_support(cluster_centroids, 0.60)
                )
                truth_with_mature_track_support += int(
                    has_support(mature_centroids, 0.60)
                )
            if prediction and not target:
                false_positive_agent_frames += 1
                ego_pose = poses[name]
                ego_position = pose_position(ego_pose)
                other_robot_positions = [
                    world_to_local(
                        pose_position(poses[other_name]),
                        [
                            ego_position[0],
                            ego_position[1],
                            float(ego_pose["yaw"]),
                        ],
                    )
                    for other_name in active_names
                    if other_name != name
                ]
                predicted_centroids = [
                    track["centroid"] for track in prediction_tracks
                ]
                false_positive_with_robot_support += int(
                    any(
                        float(
                            np.min(
                                np.linalg.norm(
                                    np.asarray(other_robot_positions)
                                    - predicted_centroid,
                                    axis=1,
                                )
                            )
                        )
                        <= 0.60
                        for predicted_centroid in predicted_centroids
                    )
                    if other_robot_positions
                    else False
                )
            if previous_poses is not None:
                update_confusion(frame_confusion, prediction, target)
                tracked_frames += 1
            episode_prediction |= prediction
            episode_target |= target
            if prediction_tracks:
                predicted_ttc = min(track["ttc"] for track in prediction_tracks)
                minimum_predicted_ttc = (
                    predicted_ttc
                    if minimum_predicted_ttc is None
                    else min(minimum_predicted_ttc, predicted_ttc)
                )
            if truth_ttc is not None:
                minimum_truth_ttc = (
                    truth_ttc
                    if minimum_truth_ttc is None
                    else min(minimum_truth_ttc, truth_ttc)
                )
        previous_poses = poses

    return {
        "case": scenario["scenario_id"],
        "risk_band": risk_band(scenario),
        "pool": scenario["preset"],
        "prediction": episode_prediction,
        "target": episode_target,
        "frame_confusion": frame_confusion,
        "tracked_agent_frames": tracked_frames,
        "detected_clusters": detected_clusters,
        "mature_tracks": mature_tracks,
        "minimum_predicted_ttc_s": minimum_predicted_ttc,
        "minimum_truth_ttc_s": minimum_truth_ttc,
        "truth_agent_frames": truth_agent_frames,
        "truth_with_raw_support": truth_with_raw_support,
        "truth_with_cluster_support": truth_with_cluster_support,
        "truth_with_mature_track_support": truth_with_mature_track_support,
        "false_positive_agent_frames": false_positive_agent_frames,
        "false_positive_with_robot_support": false_positive_with_robot_support,
    }


def main():
    args = parse_args()
    if args.min_track_age < 2:
        raise ValueError("--min-track-age must be at least 2")
    payload = load_json(args.manifest)
    scenarios = {item["scenario_id"]: item for item in payload["scenarios"]}
    frames = load_frames(args.trajectory)
    selected_frames = {}
    for episode_id in sorted(frames):
        episode_frames = frames[episode_id]
        case = episode_frames[0]["case"]
        selected_frames.setdefault(case, episode_frames)
    if set(selected_frames) != set(scenarios):
        raise ValueError("Trajectory cases do not exactly cover the manifest")

    episodes = []
    for case, episode_frames in selected_frames.items():
        scenario = scenarios[case]
        episodes.append(evaluate_episode(scenario, episode_frames, args))
    result = {
        "protocol": {
            "deployable_inputs": [
                "front_velodyne_xy_points",
                "self_odometry",
                "timestamp",
            ],
            "ground_truth_only_inputs": ["other_robot_pose_history"],
            "truth_safety_distance_m": args.truth_safety_distance,
            "predicted_collision_distance_m": args.predicted_collision_distance,
            "ttc_horizon_s": args.ttc_horizon,
            "max_encounter_distance_m": args.max_encounter_distance,
            "min_track_age": args.min_track_age,
            "association_distance_m": args.association_distance,
            "dynamic_speed_deadband_mps": args.dynamic_speed_deadband,
        },
        "frame_metrics": combine_confusions(episodes, "frame_confusion"),
        "episode_metrics": summarize_episodes(episodes),
        "observability_diagnostic": {
            "truth_agent_frames": sum(
                item["truth_agent_frames"] for item in episodes
            ),
            "raw_support_rate": sum(
                item["truth_with_raw_support"] for item in episodes
            )
            / max(sum(item["truth_agent_frames"] for item in episodes), 1),
            "cluster_support_rate": sum(
                item["truth_with_cluster_support"] for item in episodes
            )
            / max(sum(item["truth_agent_frames"] for item in episodes), 1),
            "mature_track_support_rate": sum(
                item["truth_with_mature_track_support"] for item in episodes
            )
            / max(sum(item["truth_agent_frames"] for item in episodes), 1),
            "false_positive_agent_frames": sum(
                item["false_positive_agent_frames"] for item in episodes
            ),
            "false_positive_robot_support_rate": sum(
                item["false_positive_with_robot_support"] for item in episodes
            )
            / max(
                sum(item["false_positive_agent_frames"] for item in episodes),
                1,
            ),
        },
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
                "frame_metrics": result["frame_metrics"],
                "episode_metrics": result["episode_metrics"],
                "observability_diagnostic": result["observability_diagnostic"],
                "by_risk_band": result["by_risk_band"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
