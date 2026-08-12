#!/usr/bin/env python3
"""Build leakage-audited multi-conflict views for the E2 interaction Actor."""
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

from build_g11_a1_gate_views import write_gzip_json
from scenario_manifests import AGENT_NAMES, load_manifest_dataset, validate_manifest_scenarios


DEFAULT_ROOT = (
    PROJECT_ROOT
    / "experiments/03_保留专门化/02_论文主线/datasets/fixed_v1"
)
DEFAULT_OUTPUT = DEFAULT_ROOT / "views/ie2_multi_conflict_v1"
TRAIN_QUOTAS = {
    ("standard", "edge1"): 480,
    ("dense", "edge1"): 480,
    ("standard", "edge2"): 180,
    ("dense", "edge2"): 540,
    ("standard", "edge3plus"): 60,
    ("dense", "edge3plus"): 660,
}
VALIDATION_MULTI_QUOTAS = {"edge2": 30, "edge3plus": 30}
TRAIN_SCHEDULE = ("edge1", "edge2", "edge1", "edge3plus", "edge1",
                  "edge2", "edge1", "edge3plus", "edge2", "edge3plus")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(PROJECT_ROOT))


def topology(scenario):
    edges = int(scenario["metrics"]["conflict_edge_count"])
    if edges == 1:
        return "edge1"
    if edges == 2:
        return "edge2"
    if edges >= 3:
        return "edge3plus"
    return "zero"


def rank(scenario, seed, stratum):
    value = f"{seed}:{stratum}:{scenario['scenario_id']}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load(path):
    payload = load_manifest_dataset(path)
    scenarios = validate_manifest_scenarios(payload["scenarios"], AGENT_NAMES)
    return payload, scenarios


def record(path, payload, scenarios):
    return {
        "path": relative(path),
        "dataset_id": payload.get("dataset_id"),
        "split": payload.get("split"),
        "sha256": sha256_file(path),
        "scenarios": len(scenarios),
    }


def non_train_exclusions(dataset_root):
    """Exclude every frozen non-train view, including train-derived holdouts."""
    paths = []
    for path in dataset_root.rglob("*"):
        if "trash" in path.parts or not path.is_file():
            continue
        if not (path.suffix == ".json" or path.name.endswith(".json.gz")):
            continue
        try:
            payload, scenarios = load(path)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, gzip.BadGzipFile):
            continue
        if str(payload.get("split", "")).lower() != "train":
            paths.append((path, payload, scenarios))
    return paths


def select(candidates, count, seed, stratum):
    selected = sorted(candidates, key=lambda item: rank(item, seed, stratum))[:count]
    if len(selected) != count:
        raise ValueError(f"{stratum} has {len(candidates)} candidates, need {count}")
    return selected


def annotate(source, split, pool, group):
    item = copy.deepcopy(source)
    item["navigation_split"] = source.get("split")
    item["split"] = split
    item["view"] = {
        **item.get("view", {}),
        "ie2_protocol": "ie2-multi-conflict-v1",
        "ie2_pool": pool,
        "ie2_topology": group,
        "source_scenario_id": source["scenario_id"],
    }
    return item


def interleave(groups):
    indices = Counter()
    output = []
    while len(output) < sum(len(items) for items in groups.values()):
        progressed = False
        for group in TRAIN_SCHEDULE:
            if indices[group] < len(groups[group]):
                output.append(groups[group][indices[group]])
                indices[group] += 1
                progressed = True
        if not progressed:
            break
    return output


def build(args):
    heldout = non_train_exclusions(args.dataset_root)
    heldout_ids = {
        str(item["scenario_id"])
        for _, _, scenarios in heldout
        for item in scenarios
    }
    exclusion_records = [record(*entry) for entry in heldout]

    train_groups = {name: [] for name in ("edge1", "edge2", "edge3plus")}
    train_sources = []
    eligible_counts = {}
    for pool in ("standard", "dense"):
        path = args.dataset_root / pool / "train.json.gz"
        payload, scenarios = load(path)
        train_sources.append(record(path, payload, scenarios))
        eligible = [s for s in scenarios if str(s["scenario_id"]) not in heldout_ids]
        for group in train_groups:
            candidates = [s for s in eligible if topology(s) == group]
            eligible_counts[f"{pool}_{group}"] = len(candidates)
            chosen = select(
                candidates, TRAIN_QUOTAS[(pool, group)], args.seed, f"{pool}_{group}"
            )
            train_groups[group].extend(annotate(s, "train", pool, group) for s in chosen)
    for group, items in train_groups.items():
        items.sort(key=lambda item: rank(item, args.seed + 1, group))
    train = interleave(train_groups)

    strong_path = args.dataset_root / "views/strong_interaction_curriculum_v1/validation.json.gz"
    strong_payload, strong_rows = load(strong_path)
    if any(topology(item) != "edge1" for item in strong_rows):
        raise ValueError("strong interaction validation is no longer edge-1 only")
    validation = [annotate(s, "validation", "strong", "edge1") for s in strong_rows]
    validation_sources = [record(strong_path, strong_payload, strong_rows)]

    view_heldout_ids = {
        str(item["scenario_id"])
        for path, _, scenarios in heldout
        if "views" in path.parts
        for item in scenarios
    }
    multi_candidates = {name: [] for name in VALIDATION_MULTI_QUOTAS}
    for pool in ("standard", "dense"):
        path = args.dataset_root / pool / "validation.json.gz"
        payload, scenarios = load(path)
        validation_sources.append(record(path, payload, scenarios))
        for item in scenarios:
            group = topology(item)
            if group in multi_candidates and str(item["scenario_id"]) not in view_heldout_ids:
                multi_candidates[group].append((pool, item))
    for group, quota in VALIDATION_MULTI_QUOTAS.items():
        ranked = sorted(
            multi_candidates[group],
            key=lambda pair: rank(pair[1], args.seed + 2, f"validation_{group}"),
        )
        if len(ranked) < quota:
            raise ValueError(f"validation {group} has {len(ranked)} candidates, need {quota}")
        validation.extend(annotate(item, "validation", pool, group) for pool, item in ranked[:quota])

    train_ids = [str(item["scenario_id"]) for item in train]
    validation_ids = [str(item["scenario_id"]) for item in validation]
    if len(train_ids) != len(set(train_ids)) or len(validation_ids) != len(set(validation_ids)):
        raise ValueError("duplicate scenario ID in an I-E2-M view")
    if set(train_ids) & set(validation_ids):
        raise ValueError("I-E2-M train and validation overlap")
    if set(train_ids) & heldout_ids:
        raise ValueError("a frozen heldout scenario entered I-E2-M train")

    common = {
        "protocol": "ie2-multi-conflict-v1",
        "topology_groups": {"edge1": "1", "edge2": "2", "edge3plus": ">=3"},
        "selection": "sha256_rank_without_replacement",
        "selection_seed": args.seed,
        "sealed_test_read": False,
    }
    train_payload = {
        "dataset_version": 1,
        "dataset_id": "ie2-multi-conflict-v1-train",
        "split": "train",
        "view_config": {
            **common,
            "schedule": list(TRAIN_SCHEDULE),
            "quotas": {f"{pool}_{group}": count for (pool, group), count in TRAIN_QUOTAS.items()},
            "topology_counts": dict(Counter(topology(s) for s in train)),
            "eligible_counts_after_exclusions": eligible_counts,
            "excluded_unique_scenario_count": len(heldout_ids),
            "excluded_manifests": exclusion_records,
        },
        "source_manifests": train_sources,
        "scenarios": train,
    }
    validation_payload = {
        "dataset_version": 1,
        "dataset_id": "ie2-multi-conflict-v1-validation",
        "split": "validation",
        "view_config": {
            **common,
            "topology_counts": dict(Counter(topology(s) for s in validation)),
            "train_validation_id_intersection": 0,
            "note": "edge1 reuses the frozen strong internal validation; multi strata are unused navigation-validation leftovers",
        },
        "source_manifests": validation_sources,
        "scenarios": validation,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    train_path = args.output / "train.json.gz"
    validation_path = args.output / "validation.json.gz"
    write_gzip_json(train_path, train_payload)
    write_gzip_json(validation_path, validation_payload)
    return {
        "train": {"path": relative(train_path), "sha256": sha256_file(train_path), "scenarios": len(train), "topology": dict(Counter(topology(s) for s in train))},
        "validation": {"path": relative(validation_path), "sha256": sha256_file(validation_path), "scenarios": len(validation), "topology": dict(Counter(topology(s) for s in validation))},
        "train_validation_intersection": 0,
        "train_heldout_intersection": 0,
    }


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), ensure_ascii=False, indent=2))
