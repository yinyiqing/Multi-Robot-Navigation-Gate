#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT
    / "experiments/03_保留专门化/02_论文主线/datasets/fixed_v1"
)
DEFAULT_OUTPUT = (
    DEFAULT_DATASET_ROOT / "views/weak_interaction_validation_v1/validation.json.gz"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the fixed zero-conflict-edge validation view."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write(path, payload):
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as gzip_handle:
            gzip_handle.write(encoded)


def main():
    args = parse_args()
    selected = []
    sources = []
    counts = {}
    for pool in ("standard", "dense"):
        path = args.dataset_root / pool / "validation.json.gz"
        dataset = load(path)
        pool_scenarios = []
        for scenario in dataset["scenarios"]:
            if int(scenario["metrics"]["conflict_edge_count"]) != 0:
                continue
            item = dict(scenario)
            item["view"] = {
                "interaction_band": "weak",
                "source_pool": pool,
                "policy_independent": True,
            }
            pool_scenarios.append(item)
        counts[pool] = len(pool_scenarios)
        selected.extend(pool_scenarios)
        sources.append(
            {
                "pool": pool,
                "split": "validation",
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256(path),
            }
        )

    scenario_ids = [scenario["scenario_id"] for scenario in selected]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("Weak-interaction view contains duplicate scenario IDs")
    if counts != {"standard": 206, "dense": 42}:
        raise ValueError(f"Unexpected zero-edge counts: {counts}")

    payload = {
        "dataset_version": 1,
        "dataset_id": "weak-interaction-validation-v1",
        "split": "validation",
        "view_config": {
            "view_version": 1,
            "purpose": "weak_interaction_actor_comparison",
            "selection": {"conflict_edge_count": 0},
            "pool_counts": counts,
            "policy_independent": True,
        },
        "source_manifests": sources,
        "scenarios": selected,
    }
    write(args.output, payload)
    print(json.dumps({"output": str(args.output), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
