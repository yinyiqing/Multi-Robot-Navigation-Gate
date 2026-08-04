#!/usr/bin/env python3
import argparse
import copy
import gzip
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "TD3"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from probe_start_delay_feasibility import scenario_paths
from scenario_manifests import (
    AGENT_NAMES,
    GridPlanner,
    conflict_graph,
    load_manifest_dataset,
    validate_manifest_scenarios,
)


DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT
    / "experiments/03_保留专门化/02_论文主线/datasets/fixed_v1"
)
DEFAULT_OUTPUT = DEFAULT_DATASET_ROOT / "views/g11_a1_gate_v1"
POOLS = ("standard", "dense")
TOPOLOGIES = ("zero", "edge1")
SPLITS = ("train", "validation")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build leak-free G11-A1 Gate train and validation views."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-per-stratum", type=int, default=160)
    parser.add_argument("--validation-per-stratum", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--conflict-distance", type=float, default=0.9)
    parser.add_argument("--planner-resolution", type=float, default=0.15)
    parser.add_argument("--planner-clearance", type=float, default=0.24)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path):
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(Path(path).resolve())


def write_gzip_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as gzip_handle:
            gzip_handle.write(encoded)


def full_path_horizon(paths, speed):
    longest = max(
        sum(math.dist(left, right) for left, right in zip(path, path[1:]))
        for path in paths.values()
    )
    return longest / speed + 0.2


def full_path_metrics(scenario, planner, speed, conflict_distance):
    paths = scenario_paths(scenario, planner)
    if paths is None:
        raise ValueError(
            "static path reconstruction failed: %s" % scenario["scenario_id"]
        )
    return conflict_graph(
        paths,
        nominal_speed=speed,
        conflict_distance=conflict_distance,
        horizon=full_path_horizon(paths, speed),
    )


def annotate_scenario(source, split, pool, topology):
    scenario = copy.deepcopy(source)
    scenario["navigation_split"] = scenario.get("split", "train")
    scenario["split"] = split
    scenario["view"] = {
        **scenario.get("view", {}),
        "gate_protocol": "g11-a1-v1",
        "gate_pool": pool,
        "gate_topology": topology,
        "perception_pool": pool,
        "interaction_band": "weak" if topology == "zero" else "interaction",
    }
    return scenario


def select_stratum(scenarios, train_count, validation_count, seed):
    shuffled = list(scenarios)
    random.Random(seed).shuffle(shuffled)
    required = train_count + validation_count
    if len(shuffled) < required:
        raise ValueError(
            "stratum has %d scenarios, need %d" % (len(shuffled), required)
        )
    return shuffled[:train_count], shuffled[train_count:required]


def load_scenarios(path):
    payload = load_manifest_dataset(path)
    scenarios = validate_manifest_scenarios(payload["scenarios"], AGENT_NAMES)
    return payload, scenarios


def source_record(path, payload, scenarios):
    return {
        "path": relative_path(path),
        "dataset_id": payload.get("dataset_id"),
        "split": payload.get("split"),
        "sha256": sha256(path),
        "scenarios": len(scenarios),
    }


def build_views(args):
    if args.train_per_stratum < 1 or args.validation_per_stratum < 1:
        raise ValueError("split sizes must be positive")
    if min(
        args.speed,
        args.conflict_distance,
        args.planner_resolution,
        args.planner_clearance,
    ) <= 0.0:
        raise ValueError("geometry and motion parameters must be positive")

    source_paths = {
        pool: args.dataset_root / pool / "train.json.gz" for pool in POOLS
    }
    edge1_path = (
        args.dataset_root / "views/edge1_full_horizon_v1/train.json.gz"
    )
    exclusion_paths = [
        args.dataset_root / "views/robot_perception_v1/pilot_train.json.gz",
        args.dataset_root / "views/robot_perception_v1/pilot_validation.json.gz",
    ]
    source_payloads = {}
    source_scenarios = {}
    sources = []
    for name, path in {**source_paths, "edge1": edge1_path}.items():
        payload, scenarios = load_scenarios(path)
        source_payloads[name] = payload
        source_scenarios[name] = scenarios
        sources.append(source_record(path, payload, scenarios))

    excluded_ids = set()
    exclusions = []
    for path in exclusion_paths:
        payload, scenarios = load_scenarios(path)
        excluded_ids.update(scenario["scenario_id"] for scenario in scenarios)
        exclusions.append(source_record(path, payload, scenarios))

    planner = GridPlanner(args.planner_resolution, args.planner_clearance)
    available = {}
    zero_audit = {}
    for pool in POOLS:
        zero_candidates = [
            scenario
            for scenario in source_scenarios[pool]
            if int(scenario["metrics"]["conflict_edge_count"]) == 0
            and scenario["scenario_id"] not in excluded_ids
        ]
        retained = []
        transitions = Counter()
        rejected = []
        for scenario in zero_candidates:
            metrics = full_path_metrics(
                scenario, planner, args.speed, args.conflict_distance
            )
            full_edges = int(metrics["conflict_edge_count"])
            transitions[(0, full_edges)] += 1
            if full_edges == 0:
                retained.append(scenario)
            else:
                rejected.append(scenario["scenario_id"])
        available[(pool, "zero")] = retained
        zero_audit[pool] = {
            "stored_zero_candidates_after_exclusions": len(zero_candidates),
            "retained_full_path_zero": len(retained),
            "transitions": {
                "%d->%d" % key: value for key, value in sorted(transitions.items())
            },
            "rejected_scenario_ids": rejected,
        }
        available[(pool, "edge1")] = [
            scenario
            for scenario in source_scenarios["edge1"]
            if scenario["preset"] == pool
            and scenario["scenario_id"] not in excluded_ids
        ]

    selected = {split: [] for split in SPLITS}
    counts = {}
    for pool_index, pool in enumerate(POOLS):
        for topology_index, topology in enumerate(TOPOLOGIES):
            stratum = available[(pool, topology)]
            train, validation = select_stratum(
                stratum,
                args.train_per_stratum,
                args.validation_per_stratum,
                args.seed + pool_index * 1009 + topology_index * 9173,
            )
            counts["%s_%s" % (pool, topology)] = {
                "available": len(stratum),
                "train": len(train),
                "validation": len(validation),
            }
            selected["train"].extend(
                annotate_scenario(item, "train", pool, topology) for item in train
            )
            selected["validation"].extend(
                annotate_scenario(item, "validation", pool, topology)
                for item in validation
            )

    for split_index, split in enumerate(SPLITS):
        random.Random(args.seed + 50000 + split_index).shuffle(selected[split])
    train_ids = {item["scenario_id"] for item in selected["train"]}
    validation_ids = {item["scenario_id"] for item in selected["validation"]}
    overlap = train_ids & validation_ids
    if overlap:
        raise ValueError("Gate train/validation overlap: %s" % sorted(overlap)[:5])
    if (train_ids | validation_ids) & excluded_ids:
        raise ValueError("excluded pilot scenario entered G11-A1")

    config = {
        "protocol": "g11-a1-v1",
        "source_navigation_split": "train",
        "seed": args.seed,
        "train_per_stratum": args.train_per_stratum,
        "validation_per_stratum": args.validation_per_stratum,
        "strata": ["scenario_pool", "full_path_zero_vs_corrected_edge1"],
        "sealed_test_read": False,
        "old_pilot_exclusions": exclusions,
        "old_pilot_excluded_scenario_count": len(excluded_ids),
        "zero_full_path_audit": {
            "parameters": {
                "speed_mps": args.speed,
                "conflict_distance_m": args.conflict_distance,
                "full_horizon": "longest reconstructed path duration + 0.2 s",
                "planner_resolution_m": args.planner_resolution,
                "planner_clearance_m": args.planner_clearance,
            },
            "pools": zero_audit,
        },
        "edge1_source_is_full_path_corrected": True,
    }
    outputs = {}
    for split in SPLITS:
        payload = {
            "dataset_version": 1,
            "dataset_id": "g11-a1-gate-v1-%s" % split,
            "split": split,
            "view_config": config,
            "source_manifests": sources,
            "stratum_counts": counts,
            "scenarios": selected[split],
        }
        path = args.output / (split + ".json.gz")
        write_gzip_json(path, payload)
        outputs[split] = {
            "path": relative_path(path),
            "sha256": sha256(path),
            "scenarios": len(selected[split]),
        }
    return {"outputs": outputs, "strata": counts, "zero_audit": zero_audit}


def main():
    args = parse_args()
    result = build_views(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
