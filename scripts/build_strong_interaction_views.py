#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "TD3"))

from scenario_manifests import AGENT_NAMES, load_manifest_dataset, validate_manifest_scenarios


RISK_BANDS = (
    ("deep", 0.0, 0.4),
    ("close", 0.4, 0.6),
    ("margin", 0.6, 0.9),
)
TRAIN_COUNTS = {"deep": 512, "close": 128, "margin": 128}
VALIDATION_COUNTS = {"deep": 60, "close": 40, "margin": 40}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build fixed strong-interaction train and validation views."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments/04_保留专门化/05_论文主线/datasets/fixed_v1"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def risk_band(scenario):
    metrics = scenario.get("metrics", {})
    if metrics.get("conflict_edge_count") != 1:
        return None
    separation = float(metrics["min_synchronized_path_separation_m"])
    for name, lower, upper in RISK_BANDS:
        if lower <= separation < upper:
            return name
    return None


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_gzip_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as gzip_handle:
            gzip_handle.write(encoded)


def annotate(scenario, band):
    annotated = dict(scenario)
    annotated["view"] = {
        "interaction_band": band,
        "policy_independent": True,
        "source": "synchronized_nominal_path_separation",
    }
    return annotated


def sample_balanced(grouped, counts, seed):
    selected = []
    for band_index, (band, _, _) in enumerate(RISK_BANDS):
        by_pool = grouped[band]
        target = counts[band]
        base = target // 2
        remainder = target - 2 * base
        for pool_index, pool in enumerate(("standard", "dense")):
            candidates = list(by_pool[pool])
            random.Random(seed + band_index * 100 + pool_index).shuffle(candidates)
            take = base + (remainder if pool == "dense" else 0)
            if len(candidates) < take:
                raise ValueError(
                    f"{band}/{pool} has {len(candidates)} scenarios, need {take}"
                )
            selected.extend(annotate(item, band) for item in candidates[:take])
    random.Random(seed + 10000).shuffle(selected)
    return selected


def main():
    args = parse_args()
    grouped = {
        split: {
            band: {pool: [] for pool in ("standard", "dense")}
            for band, _, _ in RISK_BANDS
        }
        for split in ("train", "validation")
    }
    source_records = []
    for split in ("train", "validation"):
        for pool in ("standard", "dense"):
            path = args.dataset_root / pool / f"{split}.json.gz"
            payload = load_manifest_dataset(path)
            scenarios = validate_manifest_scenarios(payload["scenarios"], AGENT_NAMES)
            for scenario in scenarios:
                band = risk_band(scenario)
                if band is not None:
                    grouped[split][band][pool].append(scenario)
            source_records.append(
                {
                    "pool": pool,
                    "split": split,
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "sha256": sha256(path),
                }
            )

    train = sample_balanced(grouped["train"], TRAIN_COUNTS, args.seed)
    validation = sample_balanced(
        grouped["validation"], VALIDATION_COUNTS, args.seed + 5000
    )
    thresholds = {
        name: {"min_inclusive_m": lower, "max_exclusive_m": upper}
        for name, lower, upper in RISK_BANDS
    }
    common = {
        "view_version": 1,
        "purpose": "strong_interaction_expert_pilot",
        "conflict_edge_count": 1,
        "risk_bands": thresholds,
        "policy_independent": True,
        "seed": args.seed,
    }
    for split, scenarios, counts in (
        ("train", train, TRAIN_COUNTS),
        ("validation", validation, VALIDATION_COUNTS),
    ):
        payload = {
            "dataset_version": 1,
            "dataset_id": f"strong-interaction-{split}-v1",
            "split": split,
            "view_config": {**common, "band_counts": counts},
            "source_manifests": source_records,
            "scenarios": scenarios,
        }
        write_gzip_json(args.output / f"{split}.json.gz", payload)

    summary = {
        "train": {band: sum(s["view"]["interaction_band"] == band for s in train) for band in TRAIN_COUNTS},
        "validation": {
            band: sum(s["view"]["interaction_band"] == band for s in validation)
            for band in VALIDATION_COUNTS
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
