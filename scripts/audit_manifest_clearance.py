#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "TD3"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from probe_start_delay_feasibility import scenario_paths
from scenario_geometry import has_map_clearance
from scenario_manifests import GridPlanner, load_manifest_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replan manifest tasks with a specified static clearance."
    )
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--clearance", type=float, default=0.267)
    parser.add_argument("--resolution", type=float, default=0.15)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def audit(path, planner, clearance):
    payload = load_manifest_dataset(path)
    failed = []
    endpoint_clearance_failed = []
    search_failed_with_valid_endpoints = []
    for scenario in payload["scenarios"]:
        scenario_id = scenario["scenario_id"]
        endpoints_valid = all(
            has_map_clearance(agent[key], clearance)
            for agent in scenario["agents"].values()
            for key in ("start", "goal")
        )
        if not endpoints_valid:
            endpoint_clearance_failed.append(scenario_id)
        if scenario_paths(scenario, planner) is None:
            failed.append(scenario_id)
            if endpoints_valid:
                search_failed_with_valid_endpoints.append(scenario_id)
    total = len(payload["scenarios"])
    return {
        "path": str(path),
        "dataset_id": payload.get("dataset_id"),
        "split": payload.get("split"),
        "scenario_count": total,
        "all_agents_have_static_path": total - len(failed),
        "failed_scenarios": len(failed),
        "endpoint_clearance_failed": len(endpoint_clearance_failed),
        "search_failed_with_valid_endpoints": len(
            search_failed_with_valid_endpoints
        ),
        "failed_scenario_ids": failed,
        "endpoint_clearance_failed_ids": endpoint_clearance_failed,
        "search_failed_with_valid_endpoint_ids": search_failed_with_valid_endpoints,
    }


def main():
    args = parse_args()
    if args.clearance <= 0.0 or args.resolution <= 0.0:
        raise ValueError("clearance and resolution must be positive")
    planner = GridPlanner(args.resolution, args.clearance)
    report = {
        "protocol": "manifest-static-clearance-audit-v1",
        "clearance_m": args.clearance,
        "resolution_m": args.resolution,
        "limitations": [
            "Circular clearance requires rotation-in-place clearance and is conservative for an oriented nonholonomic footprint.",
            "Gazebo reset validation is stronger evidence for collision-free start poses than this circular endpoint check.",
            "A failed circular replan does not prove that the differential-drive task is infeasible.",
        ],
        "manifests": [
            audit(path, planner, args.clearance) for path in args.manifests
        ],
    }
    encoded = json.dumps(report, indent=2, ensure_ascii=True) + "\n"
    print(encoded, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
