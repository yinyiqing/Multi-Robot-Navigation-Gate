#!/usr/bin/env python3
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "TD3"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from probe_start_delay_feasibility import scenario_paths
from scenario_manifests import GridPlanner, conflict_graph, load_manifest_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare stored 8-second conflict labels with full-path labels."
    )
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--conflict-distance", type=float, default=0.9)
    parser.add_argument("--planner-resolution", type=float, default=0.15)
    parser.add_argument("--planner-clearance", type=float, default=0.24)
    return parser.parse_args()


def full_path_horizon(paths, speed):
    longest = max(
        sum(math.dist(left, right) for left, right in zip(path, path[1:]))
        for path in paths.values()
    )
    return longest / speed + 0.2


def audit_manifest(path, planner, speed, conflict_distance):
    payload = load_manifest_dataset(path)
    rows = []
    transitions = Counter()
    leaked_multi = []
    missing_edges = []
    for scenario in payload["scenarios"]:
        paths = scenario_paths(scenario, planner)
        if paths is None:
            raise ValueError(f"Static path reconstruction failed: {scenario['scenario_id']}")
        stored = scenario["metrics"]
        complete = conflict_graph(
            paths,
            nominal_speed=speed,
            conflict_distance=conflict_distance,
            horizon=full_path_horizon(paths, speed),
        )
        stored_edges = int(stored["conflict_edge_count"])
        complete_edges = int(complete["conflict_edge_count"])
        transitions[(stored_edges, complete_edges)] += 1
        if complete_edges > stored_edges:
            missing_edges.append(scenario["scenario_id"])
        if stored_edges <= 1 and complete_edges >= 2:
            leaked_multi.append(scenario["scenario_id"])
        rows.append(
            {
                "scenario_id": scenario["scenario_id"],
                "stored_conflict_edges": stored_edges,
                "full_path_conflict_edges": complete_edges,
                "stored_max_degree": int(stored["max_conflict_degree"]),
                "full_path_max_degree": int(complete["max_conflict_degree"]),
                "stored_simultaneous": int(stored["simultaneous_conflict_count"]),
                "full_path_simultaneous": int(
                    complete["simultaneous_conflict_count"]
                ),
                "full_path_horizon_s": full_path_horizon(paths, speed),
            }
        )
    return {
        "path": str(path),
        "dataset_id": payload.get("dataset_id"),
        "split": payload.get("split"),
        "scenario_count": len(rows),
        "stored_to_full_edge_transitions": {
            f"{left}->{right}": count
            for (left, right), count in sorted(transitions.items())
        },
        "scenarios_with_missing_edges": len(missing_edges),
        "stored_0_or_1_becoming_multi_edge": len(leaked_multi),
        "missing_edge_scenario_ids": missing_edges,
        "leaked_multi_edge_scenario_ids": leaked_multi,
        "rows": rows,
    }


def main():
    args = parse_args()
    if min(
        args.speed,
        args.conflict_distance,
        args.planner_resolution,
        args.planner_clearance,
    ) <= 0.0:
        raise ValueError("all geometry and motion parameters must be positive")
    planner = GridPlanner(args.planner_resolution, args.planner_clearance)
    manifests = [
        audit_manifest(
            path, planner, args.speed, args.conflict_distance
        )
        for path in args.manifests
    ]
    report = {
        "protocol": "manifest-full-path-conflict-horizon-audit-v1",
        "parameters": {
            "speed_mps": args.speed,
            "conflict_distance_m": args.conflict_distance,
            "stored_horizon_s": 8.0,
            "full_horizon": "longest reconstructed path duration + 0.2 s",
            "planner_resolution_m": args.planner_resolution,
            "planner_clearance_m": args.planner_clearance,
        },
        "manifests": manifests,
    }
    encoded = json.dumps(report, indent=2, ensure_ascii=True) + "\n"
    print(encoded, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
