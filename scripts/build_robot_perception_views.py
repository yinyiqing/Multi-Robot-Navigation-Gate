#!/usr/bin/env python3
import argparse
import copy
import gzip
import hashlib
import json
import math
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "TD3"))

from scenario_manifests import AGENT_NAMES, load_manifest_dataset, validate_manifest_scenarios


STRATA = (
    ("standard", "weak"),
    ("standard", "interaction"),
    ("dense", "weak"),
    ("dense", "interaction"),
)
SPLITS = ("train", "validation", "test")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split navigation-train scenarios for robot-perception development."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments/03_保留专门化/02_论文主线/datasets/fixed_v1"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/robot_perception_v1"
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_sizes(count, ratios):
    if count < 0 or len(ratios) != 3 or not math.isclose(sum(ratios), 1.0):
        raise ValueError("split count or ratios are invalid")
    raw = [count * ratio for ratio in ratios]
    sizes = [int(math.floor(value)) for value in raw]
    remainder = count - sum(sizes)
    order = sorted(range(3), key=lambda index: (-(raw[index] - sizes[index]), index))
    for index in order[:remainder]:
        sizes[index] += 1
    return tuple(sizes)


def interaction_band(scenario):
    edges = int(scenario.get("metrics", {}).get("conflict_edge_count", -1))
    if edges < 0:
        raise ValueError("scenario is missing conflict_edge_count")
    return "weak" if edges == 0 else "interaction"


def write_gzip_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as gzip_handle:
            gzip_handle.write(encoded)


def build_views(dataset_root, seed=20260727, train_ratio=0.8, validation_ratio=0.1):
    test_ratio = 1.0 - train_ratio - validation_ratio
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(ratio <= 0.0 for ratio in ratios):
        raise ValueError("all perception split ratios must be positive")

    by_stratum = {stratum: [] for stratum in STRATA}
    source_records = []
    for pool in ("standard", "dense"):
        path = Path(dataset_root) / pool / "train.json.gz"
        payload = load_manifest_dataset(path)
        scenarios = validate_manifest_scenarios(payload["scenarios"], AGENT_NAMES)
        for scenario in scenarios:
            by_stratum[(pool, interaction_band(scenario))].append(scenario)
        source_records.append(
            {
                "pool": pool,
                "navigation_split": "train",
                "path": str(path.relative_to(PROJECT_ROOT)),
                "dataset_id": payload.get("dataset_id"),
                "sha256": sha256(path),
                "scenarios": len(scenarios),
            }
        )

    selected = {split: [] for split in SPLITS}
    stratum_counts = {}
    for stratum_index, (pool, band) in enumerate(STRATA):
        scenarios = list(by_stratum[(pool, band)])
        random.Random(seed + stratum_index * 1009).shuffle(scenarios)
        sizes = split_sizes(len(scenarios), ratios)
        stratum_counts[f"{pool}_{band}"] = dict(zip(SPLITS, sizes))
        start = 0
        for split, size in zip(SPLITS, sizes):
            for source in scenarios[start : start + size]:
                scenario = copy.deepcopy(source)
                scenario["navigation_split"] = scenario.get("split", "train")
                scenario["split"] = split
                scenario["view"] = {
                    **scenario.get("view", {}),
                    "perception_pool": pool,
                    "interaction_band": band,
                }
                selected[split].append(scenario)
            start += size

    for split_index, split in enumerate(SPLITS):
        random.Random(seed + 10000 + split_index).shuffle(selected[split])
    return selected, source_records, stratum_counts


def main():
    args = parse_args()
    selected, sources, stratum_counts = build_views(
        args.dataset_root,
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
    )
    config = {
        "source_navigation_split": "train",
        "train_ratio": args.train_ratio,
        "validation_ratio": args.validation_ratio,
        "test_ratio": 1.0 - args.train_ratio - args.validation_ratio,
        "shuffle_seed": args.seed,
        "stratification": ["scenario_pool", "conflict_edge_count_zero_vs_positive"],
        "test_is_sealed": True,
    }
    output_summary = {}
    for split in SPLITS:
        path = args.output / f"{split}.json.gz"
        payload = {
            "dataset_version": 1,
            "dataset_id": f"robot-perception-v1-{split}",
            "split": split,
            "view_config": config,
            "source_manifests": sources,
            "stratum_counts": stratum_counts,
            "scenarios": selected[split],
        }
        write_gzip_json(path, payload)
        output_summary[split] = {"path": str(path), "scenarios": len(selected[split])}
    print(json.dumps({"outputs": output_summary, "strata": stratum_counts}, indent=2))


if __name__ == "__main__":
    main()
