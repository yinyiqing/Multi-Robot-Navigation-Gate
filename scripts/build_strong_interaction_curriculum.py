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
STAGE_COUNTS = {
    "stage1": {"deep": 0, "close": 256, "margin": 128},
    "stage2": {"deep": 256, "close": 256, "margin": 128},
    "stage3": {"deep": 512, "close": 128, "margin": 128},
}
VALIDATION_COUNTS = {"deep": 60, "close": 40, "margin": 40}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a fixed close-to-deep interaction curriculum."
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


def ordered_candidates(grouped, seed):
    ordered = {band: {} for band, _, _ in RISK_BANDS}
    for band_index, (band, _, _) in enumerate(RISK_BANDS):
        for pool_index, pool in enumerate(("standard", "dense")):
            values = list(grouped[band][pool])
            random.Random(seed + band_index * 100 + pool_index).shuffle(values)
            ordered[band][pool] = values
    return ordered


def select(ordered, counts, stage, seed):
    selected = []
    for band, target in counts.items():
        per_pool = target // 2
        for pool in ("standard", "dense"):
            values = ordered[band][pool]
            if len(values) < per_pool:
                raise ValueError(
                    f"{band}/{pool} has {len(values)} scenarios, need {per_pool}"
                )
            for scenario in values[:per_pool]:
                item = dict(scenario)
                item["view"] = {
                    "interaction_band": band,
                    "curriculum_stage": stage,
                    "policy_independent": True,
                }
                selected.append(item)
    random.Random(seed + sum(ord(char) for char in stage)).shuffle(selected)
    return selected


def payload(dataset_id, split, scenarios, sources, config):
    return {
        "dataset_version": 1,
        "dataset_id": dataset_id,
        "split": split,
        "view_config": config,
        "source_manifests": sources,
        "scenarios": scenarios,
    }


def main():
    args = parse_args()
    grouped = {
        split: {
            band: {pool: [] for pool in ("standard", "dense")}
            for band, _, _ in RISK_BANDS
        }
        for split in ("train", "validation")
    }
    sources = []
    for split in ("train", "validation"):
        for pool in ("standard", "dense"):
            path = args.dataset_root / pool / f"{split}.json.gz"
            data = load_manifest_dataset(path)
            scenarios = validate_manifest_scenarios(data["scenarios"], AGENT_NAMES)
            for scenario in scenarios:
                band = risk_band(scenario)
                if band is not None:
                    grouped[split][band][pool].append(scenario)
            sources.append(
                {
                    "pool": pool,
                    "split": split,
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "sha256": sha256(path),
                }
            )

    train_order = ordered_candidates(grouped["train"], args.seed)
    validation_order = ordered_candidates(grouped["validation"], args.seed + 5000)
    thresholds = {
        name: {"min_inclusive_m": lower, "max_exclusive_m": upper}
        for name, lower, upper in RISK_BANDS
    }
    summary = {}
    for stage, counts in STAGE_COUNTS.items():
        scenarios = select(train_order, counts, stage, args.seed)
        config = {
            "view_version": 1,
            "purpose": "strong_interaction_curriculum",
            "curriculum_stage": stage,
            "band_counts": counts,
            "risk_bands": thresholds,
            "policy_independent": True,
            "seed": args.seed,
        }
        write_gzip_json(
            args.output / f"{stage}_train.json.gz",
            payload(f"strong-interaction-{stage}-train-v1", "train", scenarios, sources, config),
        )
        summary[stage] = counts

    validation = select(
        validation_order, VALIDATION_COUNTS, "validation", args.seed + 5000
    )
    validation_config = {
        "view_version": 1,
        "purpose": "strong_interaction_curriculum_validation",
        "band_counts": VALIDATION_COUNTS,
        "risk_bands": thresholds,
        "policy_independent": True,
        "seed": args.seed,
    }
    write_gzip_json(
        args.output / "validation.json.gz",
        payload(
            "strong-interaction-curriculum-validation-v1",
            "validation",
            validation,
            sources,
            validation_config,
        ),
    )
    summary["validation"] = VALIDATION_COUNTS
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
