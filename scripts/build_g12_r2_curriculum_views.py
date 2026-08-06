#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "TD3"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_g11_a1_gate_views import write_gzip_json
from scenario_manifests import AGENT_NAMES, load_manifest_dataset, validate_manifest_scenarios


DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT
    / "experiments/03_保留专门化/02_论文主线/datasets/fixed_v1"
)
DEFAULT_OUTPUT = DEFAULT_DATASET_ROOT / "views/g12_r2_curriculum_v1"
AGENT_COUNTS = (1, 2, 3, 5)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build fixed n1/n2/n3/n5 manifests for the G12-R2 curriculum."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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


def filtered_conflict_metrics(source_metrics, retained_names):
    retained = set(retained_names)
    edges = [
        copy.deepcopy(edge)
        for edge in source_metrics.get("conflict_edges", [])
        if set(edge.get("agents", [])) <= retained
    ]
    degrees = {name: 0 for name in retained_names}
    for edge in edges:
        for name in edge["agents"]:
            degrees[name] += 1
    edge_times = [float(edge["time_s"]) for edge in edges]
    simultaneous = 0
    for time_value in edge_times:
        simultaneous = max(
            simultaneous,
            sum(abs(other - time_value) <= 0.5 for other in edge_times),
        )
    denominator = len(retained_names) * (len(retained_names) - 1)
    return {
        "conflict_edges": edges,
        "conflict_edge_count": len(edges),
        "interaction_density": 2.0 * len(edges) / denominator if denominator else 0.0,
        "max_conflict_degree": max(degrees.values(), default=0),
        "mean_conflict_degree": (
            sum(degrees.values()) / len(degrees) if degrees else 0.0
        ),
        "earliest_conflict_time_s": min(edge_times) if edge_times else None,
        "simultaneous_conflict_count": simultaneous,
        "derived_from_source_five_agent_paths": True,
    }


def derive_scenario(source, split, agent_count):
    names = AGENT_NAMES[:agent_count]
    scenario = copy.deepcopy(source)
    source_metrics = copy.deepcopy(scenario.get("metrics", {}))
    scenario["agents"] = {name: scenario["agents"][name] for name in names}
    scenario["metrics"] = filtered_conflict_metrics(source_metrics, names)
    scenario["navigation_split"] = source.get("navigation_split", source.get("split"))
    scenario["split"] = split
    scenario["view"] = {
        **scenario.get("view", {}),
        "capacity_protocol": "g12-r2-curriculum-v1",
        "capacity_stage_agents": agent_count,
        "source_five_agent_conflict_edge_count": source_metrics.get(
            "conflict_edge_count"
        ),
    }
    return scenario


def load_source(path):
    payload = load_manifest_dataset(path)
    scenarios = validate_manifest_scenarios(payload["scenarios"], AGENT_NAMES)
    return payload, scenarios


def build_views(dataset_root, output):
    dataset_root = Path(dataset_root)
    output = Path(output)
    source_paths = {
        "train": dataset_root / "standard/train.json.gz",
        "validation": (
            dataset_root
            / "views/g12_full_scene_selection_v1/validation.json.gz"
        ),
    }
    source_payloads = {}
    source_scenarios = {}
    source_records = {}
    for split, path in source_paths.items():
        payload, scenarios = load_source(path)
        source_payloads[split] = payload
        source_scenarios[split] = scenarios
        source_records[split] = {
            "path": relative_path(path),
            "dataset_id": payload.get("dataset_id"),
            "sha256": sha256_file(path),
            "scenarios": len(scenarios),
        }

    train_ids = {item["scenario_id"] for item in source_scenarios["train"]}
    validation_ids = {
        item["scenario_id"] for item in source_scenarios["validation"]
    }
    if train_ids & validation_ids:
        raise ValueError("G12-R2 train and validation source IDs overlap")

    results = {}
    for agent_count in AGENT_COUNTS:
        stage_dir = output / ("n%d" % agent_count)
        names = AGENT_NAMES[:agent_count]
        for split in ("train", "validation"):
            scenarios = [
                derive_scenario(item, split, agent_count)
                for item in source_scenarios[split]
            ]
            validate_manifest_scenarios(scenarios, names)
            payload = {
                "dataset_version": 1,
                "dataset_id": "g12-r2-curriculum-v1-n%d-%s"
                % (agent_count, split),
                "split": split,
                "view_config": {
                    "protocol": "g12-r2-curriculum-v1",
                    "agent_count": agent_count,
                    "agent_names": names,
                    "selection": "all_source_scenarios_in_frozen_order",
                    "source_manifest": source_records[split],
                    "source_metrics_note": (
                        "conflict edges are filtered from the source five-agent "
                        "static-path graph; no scenario is selected by topology"
                    ),
                    "train_validation_id_intersection": 0,
                    "sealed_test_read": False,
                },
                "scenarios": scenarios,
            }
            path = stage_dir / (split + ".json.gz")
            write_gzip_json(path, payload)
            results["n%d_%s" % (agent_count, split)] = {
                "path": relative_path(path),
                "sha256": sha256_file(path),
                "scenarios": len(scenarios),
            }
    return results


def main():
    args = parse_args()
    print(json.dumps(build_views(args.dataset_root, args.output), indent=2))


if __name__ == "__main__":
    main()
