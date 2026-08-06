#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "TD3"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_g11_a1_gate_views import write_gzip_json
from scenario_manifests import (
    AGENT_NAMES,
    load_manifest_dataset,
    validate_manifest_scenarios,
)


DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT
    / "experiments/03_保留专门化/02_论文主线/datasets/fixed_v1"
)
DEFAULT_OUTPUT = (
    DEFAULT_DATASET_ROOT / "views/g12_full_scene_selection_v1/validation.json.gz"
)
POOLS = ("standard", "dense")
TOPOLOGIES = ("zero", "edge1", "multi")
DEFAULT_QUOTAS = {
    ("standard", "zero"): 35,
    ("standard", "edge1"): 20,
    ("standard", "multi"): 5,
    ("dense", "zero"): 5,
    ("dense", "edge1"): 20,
    ("dense", "multi"): 35,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the independent G12 full-scene selection view."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260806)
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


def source_record(path, payload, scenarios, name=None):
    record = {
        "path": relative_path(path),
        "dataset_id": payload.get("dataset_id"),
        "split": payload.get("split"),
        "sha256": sha256_file(path),
        "scenarios": len(scenarios),
    }
    if name is not None:
        record["name"] = name
    return record


def topology_name(scenario):
    edge_count = int(scenario["metrics"]["conflict_edge_count"])
    if edge_count < 0:
        raise ValueError("conflict_edge_count must be nonnegative")
    if edge_count == 0:
        return "zero"
    if edge_count == 1:
        return "edge1"
    return "multi"


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
        "capacity_protocol": "g12-full-scene-selection-v1",
        "capacity_pool": pool,
        "capacity_topology": topology,
        "g12_stratum": "%s_%s" % (pool, topology),
    }
    return scenario


def default_paths(dataset_root):
    validation = {
        pool: dataset_root / pool / "validation.json.gz" for pool in POOLS
    }
    exclusions = {
        "navigation_standard_train": dataset_root / "standard/train.json.gz",
        "navigation_dense_train": dataset_root / "dense/train.json.gz",
        "g11_c_pilot": dataset_root / "views/g11_c_pilot_v1/validation.json.gz",
        "g11_d2_admission": (
            dataset_root / "views/g11_d2_admission_v1/validation.json.gz"
        ),
        "g11_e_pilot": (
            dataset_root / "views/g11_e_edge2_generalization_v1/pilot.json.gz"
        ),
        "g11_e_confirmation": (
            dataset_root
            / "views/g11_e_edge2_generalization_v1/confirmation.json.gz"
        ),
    }
    return validation, exclusions


def build_view(args):
    if any(count < 1 for count in DEFAULT_QUOTAS.values()):
        raise ValueError("all G12 quotas must be positive")

    source_paths, exclusion_paths = default_paths(args.dataset_root)
    source_scenarios = {}
    sources = []
    for pool, path in source_paths.items():
        payload, scenarios = load_scenarios(path)
        source_scenarios[pool] = scenarios
        sources.append(source_record(path, payload, scenarios))

    excluded_ids = set()
    exclusions = []
    exclusion_id_sets = {}
    for name, path in exclusion_paths.items():
        payload, scenarios = load_scenarios(path)
        ids = {item["scenario_id"] for item in scenarios}
        exclusion_id_sets[name] = ids
        excluded_ids.update(ids)
        exclusions.append(source_record(path, payload, scenarios, name=name))

    available = {}
    candidate_counts = {}
    selected = []
    strata = {}
    source_id_sets = {
        pool: {item["scenario_id"] for item in scenarios}
        for pool, scenarios in source_scenarios.items()
    }
    if source_id_sets["standard"] & source_id_sets["dense"]:
        raise ValueError("standard and dense validation IDs overlap")

    for pool in POOLS:
        retained = [
            item
            for item in source_scenarios[pool]
            if item["scenario_id"] not in excluded_ids
        ]
        candidate_counts[pool] = dict(Counter(topology_name(item) for item in retained))
        for topology in TOPOLOGIES:
            candidates = [item for item in retained if topology_name(item) == topology]
            available[(pool, topology)] = candidates
            quota = DEFAULT_QUOTAS[(pool, topology)]
            chosen = select_ranked(candidates, quota, args.seed, pool, topology)
            selected.extend(annotate(item, pool, topology) for item in chosen)
            strata["%s_%s" % (pool, topology)] = {
                "eligible": len(candidates),
                "selected": len(chosen),
            }

    selected.sort(key=lambda item: scenario_rank(item, args.seed + 1, "all", "all"))
    selected_ids = {item["scenario_id"] for item in selected}
    if len(selected_ids) != len(selected):
        raise ValueError("G12 selection contains duplicate scenario IDs")
    if selected_ids & excluded_ids:
        raise ValueError("excluded scenario entered G12 selection")

    exclusion_intersections = {
        name: len(selected_ids & ids) for name, ids in exclusion_id_sets.items()
    }
    payload = {
        "dataset_version": 1,
        "dataset_id": "g12-full-scene-selection-v1-validation",
        "split": "validation",
        "view_config": {
            "protocol": "g12-full-scene-selection-v1",
            "source_navigation_split": "validation",
            "topology_source": "metrics.conflict_edge_count",
            "topology_groups": {"zero": "0", "edge1": "1", "multi": ">=2"},
            "selection": "sha256_rank_within_stratum",
            "selection_seed": args.seed,
            "quotas": {
                "%s_%s" % key: value for key, value in DEFAULT_QUOTAS.items()
            },
            "strata": strata,
            "candidate_counts_after_exclusions": candidate_counts,
            "sealed_test_read": False,
            "excluded_manifests": exclusions,
            "excluded_unique_scenario_count": len(excluded_ids),
            "selected_exclusion_intersections": exclusion_intersections,
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
        "candidate_counts_after_exclusions": candidate_counts,
        "excluded_unique_scenario_count": len(excluded_ids),
        "selected_exclusion_intersections": exclusion_intersections,
    }


def main():
    args = parse_args()
    print(json.dumps(build_view(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
