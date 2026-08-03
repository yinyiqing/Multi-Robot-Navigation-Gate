#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "TD3"))

from scenario_manifests import load_manifest_dataset, validate_manifest_scenarios


NUMERIC_FEATURES = (
    "conflict_edge_count",
    "max_conflict_degree",
    "simultaneous_conflict_count",
    "min_synchronized_path_separation_m",
    "mean_task_distance_m",
    "mean_path_length_m",
    "mean_path_detour_ratio",
    "min_start_clearance_m",
    "min_goal_clearance_m",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit fixed scenario quality, difficulty, and split drift."
    )
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--robot-radius", type=float, default=0.267)
    parser.add_argument("--nominal-conflict-distance", type=float, default=0.90)
    return parser.parse_args()


def quantiles(values):
    array = np.asarray(values, dtype=float)
    if not len(array):
        return None
    return {
        "min": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def edge_bucket(edge_count):
    edge_count = int(edge_count)
    if edge_count <= 0:
        return "0"
    if edge_count == 1:
        return "1"
    if edge_count == 2:
        return "2"
    return "3+"


def severity_bucket(separation, physical_collision_distance, nominal_distance):
    separation = float(separation)
    if separation < physical_collision_distance:
        return "physical_overlap_on_nominal_paths"
    if separation < nominal_distance:
        return "safety_margin_conflict"
    return "no_nominal_conflict"


def normalized_histogram(values):
    counts = Counter(values)
    total = sum(counts.values())
    return {
        str(key): value / total for key, value in sorted(counts.items())
    } if total else {}


def total_variation(left, right):
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def standardized_mean_difference(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    pooled_variance = 0.5 * (float(np.var(left)) + float(np.var(right)))
    difference = abs(float(np.mean(left)) - float(np.mean(right)))
    return difference / math.sqrt(pooled_variance) if pooled_variance > 0.0 else difference


def scenario_features(scenario):
    agents = list(scenario["agents"].values())
    task_distances = [float(agent["task_distance_m"]) for agent in agents]
    path_lengths = [float(agent["path_length_m"]) for agent in agents]
    detours = [
        path_length / task_distance
        for path_length, task_distance in zip(path_lengths, task_distances)
    ]
    metrics = scenario["metrics"]
    validity = scenario["validity"]
    return {
        "conflict_edge_count": float(metrics["conflict_edge_count"]),
        "max_conflict_degree": float(metrics["max_conflict_degree"]),
        "simultaneous_conflict_count": float(
            metrics["simultaneous_conflict_count"]
        ),
        "min_synchronized_path_separation_m": float(
            metrics["min_synchronized_path_separation_m"]
        ),
        "mean_task_distance_m": float(np.mean(task_distances)),
        "mean_path_length_m": float(np.mean(path_lengths)),
        "mean_path_detour_ratio": float(np.mean(detours)),
        "min_start_clearance_m": float(validity["min_start_clearance_m"]),
        "min_goal_clearance_m": float(validity["min_goal_clearance_m"]),
    }


def layout_digest(scenario):
    layout = {
        "preset": scenario["preset"],
        "agents": scenario["agents"],
        "boxes": scenario["boxes"],
    }
    encoded = json.dumps(layout, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def summarize_manifest(path, payload, scenarios, physical_distance, nominal_distance):
    feature_rows = [scenario_features(item) for item in scenarios]
    features = {
        name: [row[name] for row in feature_rows] for name in NUMERIC_FEATURES
    }
    edge_buckets = [edge_bucket(row["conflict_edge_count"]) for row in feature_rows]
    severity = [
        severity_bucket(
            row["min_synchronized_path_separation_m"],
            physical_distance,
            nominal_distance,
        )
        for row in feature_rows
    ]
    return {
        "path": str(path),
        "dataset_id": payload.get("dataset_id"),
        "preset": scenarios[0]["preset"],
        "split": payload.get("split"),
        "scenario_count": len(scenarios),
        "gazebo_reset_passed": sum(
            item.get("validity", {}).get("gazebo_reset") is True
            for item in scenarios
        ),
        "static_path_passed": sum(
            item.get("validity", {}).get("static_path_for_all_agents") is True
            for item in scenarios
        ),
        "edge_bucket_counts": dict(sorted(Counter(edge_buckets).items())),
        "edge_bucket_rates": normalized_histogram(edge_buckets),
        "severity_counts": dict(sorted(Counter(severity).items())),
        "severity_rates": normalized_histogram(severity),
        "feature_quantiles": {
            name: quantiles(values) for name, values in features.items()
        },
        "generator_config": payload.get("generator_config", {}),
        "_features": features,
    }


def compare_splits(train, candidate):
    feature_smd = {
        name: standardized_mean_difference(
            train["_features"][name], candidate["_features"][name]
        )
        for name in NUMERIC_FEATURES
    }
    return {
        "reference": train["split"],
        "candidate": candidate["split"],
        "edge_bucket_total_variation": total_variation(
            train["edge_bucket_rates"], candidate["edge_bucket_rates"]
        ),
        "feature_standardized_mean_difference": feature_smd,
        "max_feature_smd": max(feature_smd.values(), default=0.0),
        "features_over_0_10_smd": sorted(
            name for name, value in feature_smd.items() if value > 0.10
        ),
    }


def build_report(paths, num_agents, robot_radius, nominal_distance):
    if num_agents < 1:
        raise ValueError("num_agents must be positive")
    if robot_radius <= 0.0 or nominal_distance <= 2.0 * robot_radius:
        raise ValueError("conflict distance must exceed the robot diameter")
    physical_distance = 2.0 * robot_radius
    all_ids = set()
    all_layouts = set()
    duplicate_ids = []
    duplicate_layouts = []
    summaries = []
    agent_names = tuple(f"r{index}" for index in range(1, num_agents + 1))

    for path in paths:
        payload = load_manifest_dataset(path)
        scenarios = validate_manifest_scenarios(payload["scenarios"], agent_names)
        for scenario in scenarios:
            scenario_id = scenario["scenario_id"]
            if scenario_id in all_ids:
                duplicate_ids.append(scenario_id)
            all_ids.add(scenario_id)
            digest = layout_digest(scenario)
            if digest in all_layouts:
                duplicate_layouts.append(scenario_id)
            all_layouts.add(digest)
        summaries.append(
            summarize_manifest(
                path, payload, scenarios, physical_distance, nominal_distance
            )
        )

    by_preset = defaultdict(list)
    for summary in summaries:
        by_preset[summary["preset"]].append(summary)
    comparisons = []
    generator_mismatches = []
    for preset, items in sorted(by_preset.items()):
        train = next((item for item in items if item["split"] == "train"), None)
        if train is None:
            continue
        for item in items:
            if item is train:
                continue
            comparisons.append({"preset": preset, **compare_splits(train, item)})
            if item["generator_config"] != train["generator_config"]:
                generator_mismatches.append(
                    {"preset": preset, "split": item["split"]}
                )

    planner_clearances = sorted(
        {
            float(item["generator_config"].get("robot_map_clearance", math.nan))
            for item in summaries
        }
    )
    issues = []
    if duplicate_ids:
        issues.append("scenario_id_overlap_across_splits")
    if duplicate_layouts:
        issues.append("exact_layout_duplicate_across_splits")
    if generator_mismatches:
        issues.append("generator_config_drift_within_preset")
    if any(value < robot_radius for value in planner_clearances):
        issues.append("planner_clearance_below_robot_radius")
    if any(item["gazebo_reset_passed"] != item["scenario_count"] for item in summaries):
        issues.append("gazebo_reset_not_passed")
    if any(item["static_path_passed"] != item["scenario_count"] for item in summaries):
        issues.append("missing_static_path")

    public_summaries = []
    for summary in summaries:
        public_summaries.append(
            {key: value for key, value in summary.items() if key != "_features"}
        )
    return {
        "protocol": "fixed-v1-scenario-quality-audit-v1",
        "scenario_count": sum(item["scenario_count"] for item in summaries),
        "unique_scenario_ids": len(all_ids),
        "unique_layouts": len(all_layouts),
        "assumptions": {
            "robot_radius_m": robot_radius,
            "physical_pair_distance_m": physical_distance,
            "nominal_conflict_distance_m": nominal_distance,
            "planner_clearance_m": planner_clearances,
        },
        "issues": issues,
        "duplicate_scenario_ids": duplicate_ids,
        "duplicate_layout_scenario_ids": duplicate_layouts,
        "generator_config_mismatches": generator_mismatches,
        "manifests": public_summaries,
        "split_comparisons": comparisons,
    }


def main():
    args = parse_args()
    report = build_report(
        args.manifests,
        args.num_agents,
        args.robot_radius,
        args.nominal_conflict_distance,
    )
    encoded = json.dumps(report, indent=2, ensure_ascii=True) + "\n"
    print(encoded, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
