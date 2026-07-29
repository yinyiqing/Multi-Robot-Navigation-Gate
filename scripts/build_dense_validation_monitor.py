#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/dense/validation.json.gz"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/"
    "dense_validation_monitor_v1/validation.json.gz"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a policy-independent monitor subset of dense validation."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--dataset-id", default="dense-validation-monitor-v1")
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


def conflict_summary(scenarios):
    edges = [int(item["metrics"]["conflict_edge_count"]) for item in scenarios]
    return {
        "scenarios": len(edges),
        "mean_conflict_edges": sum(edges) / len(edges),
        "edge_0": sum(value == 0 for value in edges),
        "edge_ge_1": sum(value >= 1 for value in edges),
        "edge_ge_2": sum(value >= 2 for value in edges),
        "edge_ge_3": sum(value >= 3 for value in edges),
    }


def main():
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be positive")

    source = load(args.source)
    scenarios = source["scenarios"]
    selected = [dict(item) for item in scenarios[:: args.stride]]
    if not selected:
        raise ValueError("Monitor selection is empty")

    scenario_ids = [item["scenario_id"] for item in selected]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("Monitor view contains duplicate scenario IDs")

    payload = {
        "dataset_version": 1,
        "dataset_id": args.dataset_id,
        "split": "validation",
        "view_config": {
            "view_version": 1,
            "purpose": "frequent_dense_actor_checkpoint_selection",
            "selection": {
                "method": "fixed_stride",
                "stride": args.stride,
                "start_index": 0,
            },
            "policy_independent": True,
            "source_summary": conflict_summary(scenarios),
            "monitor_summary": conflict_summary(selected),
        },
        "source_manifests": [
            {
                "dataset_id": source.get("dataset_id"),
                "split": source.get("split"),
                "path": str(args.source.resolve().relative_to(PROJECT_ROOT)),
                "sha256": sha256(args.source),
            }
        ],
        "scenarios": selected,
    }
    write(args.output, payload)
    print(json.dumps(payload["view_config"], ensure_ascii=False, indent=2))
    print("output:", args.output)
    print("sha256:", sha256(args.output))


if __name__ == "__main__":
    main()
