#!/usr/bin/env python3
import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "TD3"))

from lidar_cluster_tracking import cluster_points, world_to_local  # noqa: E402


FEATURE_NAMES = (
    "xy_diameter",
    "xy_aspect_ratio",
    "point_count",
    "z_min",
    "z_max",
    "z_mean",
    "z_std",
    "z_span",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit whether 3D lidar cluster shape separates robots from clutter."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot-association-distance", type=float, default=0.60)
    parser.add_argument("--max-cluster-range", type=float, default=4.0)
    return parser.parse_args()


def load_json(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def load_first_case_episodes(path):
    by_episode = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            frame = json.loads(line)
            points = frame.get("raw_lidar_points")
            if points is None:
                raise ValueError("Trajectory does not contain raw_lidar_points")
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
    calibration = set()
    evaluation = set()
    for case_ids in groups.values():
        for index, case_id in enumerate(sorted(case_ids)):
            (calibration if index % 2 == 0 else evaluation).add(case_id)
    return calibration, evaluation


def cluster_features(points, cluster):
    cluster_xyz = points[cluster["indices"]]
    xy_extent = np.ptp(cluster_xyz[:, :2], axis=0)
    shorter = max(float(np.min(xy_extent)), 1e-3)
    z_values = cluster_xyz[:, 2]
    return {
        "xy_diameter": float(cluster["diameter"]),
        "xy_aspect_ratio": float(np.max(xy_extent) / shorter),
        "point_count": float(len(cluster_xyz)),
        "z_min": float(np.min(z_values)),
        "z_max": float(np.max(z_values)),
        "z_mean": float(np.mean(z_values)),
        "z_std": float(np.std(z_values)),
        "z_span": float(np.ptp(z_values)),
    }


def extract_records(scenarios, frames_by_case, max_range, association_distance):
    records = []
    for case_id, frames in frames_by_case.items():
        scenario = scenarios[case_id]
        agent_names = list(scenario["agents"])
        for frame in frames:
            active_names = {
                name
                for index, name in enumerate(agent_names)
                if frame["active_before"][index]
            }
            for name in active_names:
                points = np.asarray(
                    frame["raw_lidar_points"][name], dtype=np.float64
                )
                if points.ndim != 2 or points.shape[1] < 3:
                    raise ValueError("Shape probe requires XYZ raw lidar points")
                pose = frame["actor_poses"][name]
                ego_pose = [pose["x"], pose["y"], pose["yaw"]]
                other_positions = []
                for other_name in active_names:
                    if other_name == name:
                        continue
                    other_pose = frame["actor_poses"][other_name]
                    other_positions.append(
                        world_to_local(
                            [other_pose["x"], other_pose["y"]], ego_pose
                        )
                    )
                for cluster in cluster_points(points):
                    if np.linalg.norm(cluster["centroid"]) > max_range:
                        continue
                    nearest_robot_distance = (
                        min(
                            float(np.linalg.norm(cluster["centroid"] - position))
                            for position in other_positions
                        )
                        if other_positions
                        else float("inf")
                    )
                    records.append(
                        {
                            "case": case_id,
                            "risk_band": risk_band(scenario),
                            "pool": scenario["preset"],
                            "is_robot": nearest_robot_distance
                            <= association_distance,
                            "nearest_robot_distance": nearest_robot_distance,
                            **cluster_features(points, cluster),
                        }
                    )
    return records


def metrics(predictions, targets):
    predictions = np.asarray(predictions, dtype=bool)
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
    }


def feature_quantiles(records, feature, target):
    values = [item[feature] for item in records if item["is_robot"] == target]
    if not values:
        return None
    return {
        str(quantile): float(np.quantile(values, quantile))
        for quantile in (0.05, 0.25, 0.5, 0.75, 0.95)
    }


def fit_stump(records, feature):
    values = np.asarray([item[feature] for item in records], dtype=np.float64)
    targets = np.asarray([item["is_robot"] for item in records], dtype=bool)
    thresholds = np.unique(np.quantile(values, np.linspace(0.0, 1.0, 101)))
    best = None
    for direction in ("at_least", "at_most"):
        for threshold in thresholds:
            predictions = (
                values >= threshold if direction == "at_least" else values <= threshold
            )
            result = metrics(predictions, targets)
            score = result["recall"] - result["false_positive_rate"]
            key = (score, result["recall"], result["precision"] or 0.0)
            if best is None or key > best[0]:
                best = (
                    key,
                    {
                        "feature": feature,
                        "direction": direction,
                        "threshold": float(threshold),
                        "calibration_metrics": result,
                    },
                )
    return best[1]


def apply_stump(records, rule):
    values = np.asarray([item[rule["feature"]] for item in records])
    predictions = (
        values >= rule["threshold"]
        if rule["direction"] == "at_least"
        else values <= rule["threshold"]
    )
    return metrics(predictions, [item["is_robot"] for item in records])


def main():
    args = parse_args()
    payload = load_json(args.manifest)
    scenarios = {item["scenario_id"]: item for item in payload["scenarios"]}
    frames_by_case = load_first_case_episodes(args.trajectory)
    if set(frames_by_case) != set(scenarios):
        raise ValueError("Trajectory cases do not exactly cover the manifest")
    calibration_cases, evaluation_cases = split_cases(scenarios)
    records = extract_records(
        scenarios,
        frames_by_case,
        args.max_cluster_range,
        args.robot_association_distance,
    )
    calibration = [item for item in records if item["case"] in calibration_cases]
    evaluation = [item for item in records if item["case"] in evaluation_cases]

    rules = []
    for feature in FEATURE_NAMES:
        rule = fit_stump(calibration, feature)
        rule["evaluation_metrics"] = apply_stump(evaluation, rule)
        rules.append(rule)
    rules.sort(
        key=lambda item: (
            item["evaluation_metrics"]["recall"]
            - item["evaluation_metrics"]["false_positive_rate"],
            item["evaluation_metrics"]["recall"],
        ),
        reverse=True,
    )

    result = {
        "protocol": {
            "calibration_cases": len(calibration_cases),
            "evaluation_cases": len(evaluation_cases),
            "robot_association_distance_m": args.robot_association_distance,
            "max_cluster_range_m": args.max_cluster_range,
            "features": list(FEATURE_NAMES),
        },
        "clusters": {
            "calibration": len(calibration),
            "calibration_robot": sum(item["is_robot"] for item in calibration),
            "evaluation": len(evaluation),
            "evaluation_robot": sum(item["is_robot"] for item in evaluation),
        },
        "distributions": {
            feature: {
                "robot": feature_quantiles(records, feature, True),
                "static": feature_quantiles(records, feature, False),
            }
            for feature in FEATURE_NAMES
        },
        "univariate_rules": rules,
        "best_univariate_rule": rules[0],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
