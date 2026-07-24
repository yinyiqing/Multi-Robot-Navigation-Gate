#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "TD3"))

from scenario_geometry import has_map_clearance  # noqa: E402
from scenario_manifests import (  # noqa: E402
    conflict_graph,
    pairwise_min_distance,
    validate_manifest_scenarios,
)


AGENT_NAMES = ("r1", "r2")
TOPOLOGIES = ("head_on", "crossing", "lane_swap")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build randomized two-robot interaction curriculum manifests."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-per-topology", type=int, default=30)
    parser.add_argument("--validation-per-topology", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def rotate(point, quarter_turns):
    x, y = point
    for _ in range(quarter_turns % 4):
        x, y = -y, x
    return np.asarray([x, y], dtype=np.float64)


def transformed(point, quarter_turns, center):
    return rotate(point, quarter_turns) + center


def line_has_clearance(start, goal, clearance=0.24):
    distance = float(np.linalg.norm(goal - start))
    samples = max(int(math.ceil(distance / 0.08)), 2)
    for alpha in np.linspace(0.0, 1.0, samples + 1):
        point = start + alpha * (goal - start)
        if not has_map_clearance(point, clearance):
            return False
    return True


def canonical_geometry(topology, rng):
    distance = float(rng.uniform(1.55, 1.95))
    asymmetry = float(rng.uniform(-0.18, 0.18))
    if topology == "head_on":
        lateral = float(rng.uniform(-0.18, 0.18))
        return {
            "r1": ((-distance, -lateral), (distance, lateral)),
            "r2": ((distance, lateral + asymmetry), (-distance, -lateral + asymmetry)),
        }
    if topology == "crossing":
        offset = float(rng.uniform(-0.16, 0.16))
        return {
            "r1": ((-distance, offset), (distance, -offset)),
            "r2": ((-offset + asymmetry, -distance), (offset + asymmetry, distance)),
        }
    if topology == "lane_swap":
        distance = float(rng.uniform(1.20, 1.55))
        lane = float(rng.uniform(0.62, 0.78))
        goal_lane = float(rng.uniform(0.42, 0.55))
        return {
            "r1": ((-distance, -lane), (distance, goal_lane)),
            "r2": ((-distance, lane), (distance, -goal_lane)),
        }
    raise ValueError(f"Unknown topology: {topology}")


def build_scenario(topology, generation_seed, split):
    rng = np.random.default_rng(generation_seed)
    for _ in range(1000):
        quarter_turns = int(rng.integers(0, 4))
        center = rng.uniform(-0.8, 0.8, size=2)
        canonical = canonical_geometry(topology, rng)
        starts = {
            name: transformed(values[0], quarter_turns, center)
            for name, values in canonical.items()
        }
        goals = {
            name: transformed(values[1], quarter_turns, center)
            for name, values in canonical.items()
        }
        if pairwise_min_distance(starts.values()) < 1.2:
            continue
        if pairwise_min_distance(goals.values()) < 0.8:
            continue
        if not all(
            line_has_clearance(starts[name], goals[name]) for name in AGENT_NAMES
        ):
            continue
        paths = {
            name: [tuple(starts[name]), tuple(goals[name])] for name in AGENT_NAMES
        }
        metrics = conflict_graph(paths)
        if metrics["conflict_edge_count"] != 1:
            continue
        if metrics["min_synchronized_path_separation_m"] >= 0.45:
            continue
        agents = {}
        task_distances = []
        for name in AGENT_NAMES:
            start = starts[name]
            goal = goals[name]
            task_distance = float(np.linalg.norm(goal - start))
            task_distances.append(task_distance)
            heading = math.atan2(goal[1] - start[1], goal[0] - start[0])
            heading += float(rng.uniform(-0.12, 0.12))
            agents[name] = {
                "start": [round(float(value), 6) for value in start],
                "goal": [round(float(value), 6) for value in goal],
                "heading": round(heading, 6),
                "task_distance_m": round(task_distance, 6),
                "path_length_m": round(task_distance, 6),
            }
        core = {
            "manifest_version": 1,
            "generator": "pair-interaction-curriculum-v1",
            "preset": "structured_pair_interaction",
            "generation_seed": generation_seed,
            "map_id": "TD3.world-v1",
            "num_agents": 2,
            "boxes": [],
            "agents": agents,
            "validity": {
                "static_geometry": True,
                "static_path_for_all_agents": True,
                "gazebo_reset": None,
                "min_start_clearance_m": round(
                    pairwise_min_distance(starts.values()), 6
                ),
                "min_goal_clearance_m": round(
                    pairwise_min_distance(goals.values()), 6
                ),
                "min_task_distance_m": round(min(task_distances), 6),
                "max_task_distance_m": round(max(task_distances), 6),
            },
            "metrics": metrics,
            "interaction_topology": topology,
            "split": split,
        }
        digest = hashlib.sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:12]
        core["scenario_id"] = f"pair-{topology}-{generation_seed}-{digest}"
        return core
    raise RuntimeError(f"Could not generate a valid {topology} scenario")


def make_split(split, per_topology, master_seed):
    scenarios = []
    for topology_index, topology in enumerate(TOPOLOGIES):
        for index in range(per_topology):
            generation_seed = master_seed * 100000 + topology_index * 10000 + index
            scenarios.append(build_scenario(topology, generation_seed, split))
    rng = np.random.default_rng(master_seed + 9000)
    rng.shuffle(scenarios)
    validate_manifest_scenarios(scenarios, AGENT_NAMES)
    return {
        "dataset_version": 1,
        "dataset_id": f"pair-interaction-curriculum-v1-{split}",
        "split": split,
        "generator_config": {
            "topologies": list(TOPOLOGIES),
            "scenarios_per_topology": per_topology,
            "policy_independent": True,
            "seed": master_seed,
        },
        "scenarios": scenarios,
    }


def write_gzip_json(path, payload):
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as handle:
            handle.write(encoded)


def main():
    args = parse_args()
    if args.train_per_topology < 1 or args.validation_per_topology < 1:
        raise ValueError("Scenario counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    train = make_split("train", args.train_per_topology, args.seed)
    validation = make_split(
        "validation", args.validation_per_topology, args.seed + 1
    )
    write_gzip_json(args.output / "train_candidates.json.gz", train)
    write_gzip_json(args.output / "validation_candidates.json.gz", validation)
    print(
        json.dumps(
            {
                "train": len(train["scenarios"]),
                "validation": len(validation["scenarios"]),
                "topologies": list(TOPOLOGIES),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
