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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build deterministic interaction-stratified views of fixed manifests."
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
    parser.add_argument("--conflict-edges", type=int, default=1)
    parser.add_argument("--pilot-per-pool", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260720)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matching_scenarios(payload, split, conflict_edges):
    scenarios = validate_manifest_scenarios(payload["scenarios"], AGENT_NAMES)
    return [
        scenario
        for scenario in scenarios
        if scenario.get("split") == split
        and scenario.get("metrics", {}).get("conflict_edge_count") == conflict_edges
    ]


def shuffled(items, seed):
    result = list(items)
    random.Random(seed).shuffle(result)
    return result


def write_gzip_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as gzip_handle:
            gzip_handle.write(encoded)


def build_payload(dataset_id, split, scenarios, source_records, config):
    return {
        "dataset_version": 1,
        "dataset_id": dataset_id,
        "split": split,
        "view_config": config,
        "source_manifests": source_records,
        "scenarios": scenarios,
    }


def main():
    args = parse_args()
    if args.conflict_edges < 0:
        raise ValueError("--conflict-edges must be non-negative")
    if args.pilot_per_pool < 1:
        raise ValueError("--pilot-per-pool must be positive")

    pools = ("standard", "dense")
    sources = {}
    source_records = []
    for pool in pools:
        sources[pool] = {}
        for split in ("train", "validation"):
            path = args.dataset_root / pool / f"{split}.json.gz"
            payload = load_manifest_dataset(path)
            matches = matching_scenarios(payload, split, args.conflict_edges)
            sources[pool][split] = shuffled(
                matches,
                args.seed + (0 if pool == "standard" else 100) + (0 if split == "train" else 10),
            )
            source_records.append(
                {
                    "pool": pool,
                    "split": split,
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "dataset_id": payload.get("dataset_id"),
                    "sha256": sha256(path),
                    "matching_scenarios": len(matches),
                }
            )

    pilot_parts = []
    for pool in pools:
        available = sources[pool]["train"]
        if len(available) < args.pilot_per_pool:
            raise ValueError(
                f"{pool}/train has only {len(available)} matching scenarios, "
                f"need {args.pilot_per_pool}"
            )
        pilot_parts.extend(available[: args.pilot_per_pool])
    pilot_train = shuffled(pilot_parts, args.seed + 1000)
    validation = shuffled(
        sources["standard"]["validation"] + sources["dense"]["validation"],
        args.seed + 2000,
    )

    config = {
        "conflict_edge_count": args.conflict_edges,
        "pilot_per_pool": args.pilot_per_pool,
        "shuffle_seed": args.seed,
        "policy_independent": True,
    }
    train_payload = build_payload(
        f"interaction-edge{args.conflict_edges}-pilot-train-v1",
        "train",
        pilot_train,
        source_records,
        config,
    )
    validation_payload = build_payload(
        f"interaction-edge{args.conflict_edges}-validation-v1",
        "validation",
        validation,
        source_records,
        config,
    )
    train_path = args.output / "train.json.gz"
    validation_path = args.output / "validation.json.gz"
    write_gzip_json(train_path, train_payload)
    write_gzip_json(validation_path, validation_payload)

    print(
        json.dumps(
            {
                "train": {"path": str(train_path), "scenarios": len(pilot_train)},
                "validation": {
                    "path": str(validation_path),
                    "scenarios": len(validation),
                },
                "source_counts": {
                    pool: {
                        split: len(sources[pool][split])
                        for split in ("train", "validation")
                    }
                    for pool in pools
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
