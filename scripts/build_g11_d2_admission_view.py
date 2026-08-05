#!/usr/bin/env python3
import argparse
import copy
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "TD3"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_g11_a1_gate_views import full_path_metrics, write_gzip_json
from scenario_manifests import (
    AGENT_NAMES,
    GridPlanner,
    load_manifest_dataset,
    validate_manifest_scenarios,
)


DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT
    / "experiments/03_保留专门化/02_论文主线/datasets/fixed_v1"
)
DEFAULT_OUTPUT = DEFAULT_DATASET_ROOT / "views/g11_d2_admission_v1/validation.json.gz"
POOLS = ("standard", "dense")
TOPOLOGIES = ("zero", "edge1")
DEFAULT_QUOTAS = {
    ("standard", "zero"): 65,
    ("dense", "zero"): 35,
    ("standard", "edge1"): 50,
    ("dense", "edge1"): 50,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the independent G11-D2 0-edge/edge-1 admission view."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--conflict-distance", type=float, default=0.9)
    parser.add_argument("--planner-resolution", type=float, default=0.15)
    parser.add_argument("--planner-clearance", type=float, default=0.24)
    return parser.parse_args()


def sha256_file(path):
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


def load_scenarios(path):
    payload = load_manifest_dataset(path)
    scenarios = validate_manifest_scenarios(payload["scenarios"], AGENT_NAMES)
    return payload, scenarios


def source_record(path, payload, scenarios):
    return {
        "path": relative_path(path),
        "dataset_id": payload.get("dataset_id"),
        "split": payload.get("split"),
        "sha256": sha256_file(path),
        "scenarios": len(scenarios),
    }


def scenario_rank(scenario, seed, pool, topology):
    value = "%d:%s:%s:%s" % (seed, pool, topology, scenario["scenario_id"])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def select_ranked(scenarios, count, seed, pool, topology):
    ranked = sorted(
        scenarios,
        key=lambda item: scenario_rank(item, seed, pool, topology),
    )
    if len(ranked) < count:
        raise ValueError(
            "%s/%s has %d eligible scenarios, need %d"
            % (pool, topology, len(ranked), count)
        )
    return ranked[:count]


def annotate(source, pool, topology):
    scenario = copy.deepcopy(source)
    scenario["navigation_split"] = scenario.get("split", "validation")
    scenario["split"] = "validation"
    scenario["view"] = {
        **scenario.get("view", {}),
        "gate_protocol": "g11-d2-v1",
        "gate_pool": pool,
        "gate_topology": topology,
        "g11_d2_stratum": "%s_%s" % (pool, topology),
        "perception_pool": pool,
        "interaction_band": "weak" if topology == "zero" else "interaction",
    }
    return scenario


def build_view(args):
    if any(count < 1 for count in DEFAULT_QUOTAS.values()):
        raise ValueError("all G11-D2 quotas must be positive")
    if min(
        args.speed,
        args.conflict_distance,
        args.planner_resolution,
        args.planner_clearance,
    ) <= 0.0:
        raise ValueError("geometry and motion parameters must be positive")

    source_paths = {
        pool: args.dataset_root / pool / "validation.json.gz" for pool in POOLS
    }
    edge1_path = args.dataset_root / "views/edge1_full_horizon_v1/validation.json.gz"
    exclusion_paths = {
        "g11_a1_internal_validation": (
            args.dataset_root / "views/g11_a1_gate_v1/validation.json.gz"
        ),
        "g11_c_pilot": args.dataset_root / "views/g11_c_pilot_v1/validation.json.gz",
        "old_g3_dense_monitor": (
            args.dataset_root / "views/dense_validation_monitor_v1/validation.json.gz"
        ),
    }
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
    for name, path in exclusion_paths.items():
        payload, scenarios = load_scenarios(path)
        excluded_ids.update(item["scenario_id"] for item in scenarios)
        record = source_record(path, payload, scenarios)
        record["name"] = name
        exclusions.append(record)

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

    selected = []
    strata = {}
    for pool in POOLS:
        for topology in TOPOLOGIES:
            candidates = available[(pool, topology)]
            quota = DEFAULT_QUOTAS[(pool, topology)]
            chosen = select_ranked(
                candidates, quota, args.seed, pool, topology
            )
            selected.extend(annotate(item, pool, topology) for item in chosen)
            strata["%s_%s" % (pool, topology)] = {
                "eligible": len(candidates),
                "selected": len(chosen),
            }
    selected.sort(key=lambda item: scenario_rank(item, args.seed + 1, "all", "all"))
    selected_ids = [item["scenario_id"] for item in selected]
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("G11-D2 selection contains duplicate scenario IDs")
    if set(selected_ids) & excluded_ids:
        raise ValueError("excluded scenario entered G11-D2")

    payload = {
        "dataset_version": 1,
        "dataset_id": "g11-d2-independent-admission-v1-validation",
        "split": "validation",
        "view_config": {
            "protocol": "g11-d2-v1",
            "source_navigation_split": "validation",
            "selection": "sha256_rank_within_stratum",
            "selection_seed": args.seed,
            "quotas": {
                "%s_%s" % key: value for key, value in DEFAULT_QUOTAS.items()
            },
            "strata": strata,
            "sealed_test_read": False,
            "excluded_manifests": exclusions,
            "excluded_unique_scenario_count": len(excluded_ids),
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
        },
        "source_manifests": sources,
        "scenarios": selected,
    }
    write_gzip_json(args.output, payload)
    return {
        "output": relative_path(args.output),
        "sha256": sha256_file(args.output),
        "scenarios": len(selected),
        "strata": strata,
        "zero_audit": zero_audit,
        "excluded_unique_scenario_count": len(excluded_ids),
    }


def main():
    args = parse_args()
    result = build_view(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
