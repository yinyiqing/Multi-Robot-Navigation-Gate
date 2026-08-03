#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT
    / "experiments/03_保留专门化/02_论文主线/datasets/fixed_v1"
)
DEFAULT_AUDIT = (
    PROJECT_ROOT
    / "experiments/03_保留专门化/02_论文主线/results/04_Gate前置验证/"
    "20260803_全场景数据质量诊断/full_horizon_edge1_audit.json.gz"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a leak-free edge-1 view from the full-path conflict audit."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_DATASET_ROOT / "views/edge1_pilot",
    )
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATASET_ROOT / "views/edge1_full_horizon_v1",
    )
    parser.add_argument("--monitor-per-pool", type=int, default=25)
    parser.add_argument("--monitor-seed", type=int, default=20260803)
    return parser.parse_args()


def load_json(path):
    opener = gzip.open if path.suffix == ".gz" else Path.open
    kwargs = {"encoding": "utf-8"} if path.suffix == ".gz" else {}
    with opener(path, "rt", **kwargs) as handle:
        return json.load(handle)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path):
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def write_gzip_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as gzip_handle:
            gzip_handle.write(encoded)


def is_pure_edge1(row):
    return (
        row["full_path_conflict_edges"] == 1
        and row["full_path_max_degree"] == 1
        and row["full_path_simultaneous"] == 1
    )


def audit_rows_by_split(audit):
    result = {}
    for manifest in audit.get("manifests", []):
        split = manifest.get("split")
        if split in result:
            raise ValueError(f"duplicate audit manifest for split {split!r}")
        rows = manifest.get("rows", [])
        indexed = {row["scenario_id"]: row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError(f"duplicate scenario IDs in {split!r} audit rows")
        result[split] = indexed
    return result


def filter_split(source_payload, rows, split):
    scenarios = source_payload.get("scenarios", [])
    source_ids = [scenario["scenario_id"] for scenario in scenarios]
    missing = sorted(set(source_ids) - set(rows))
    extra = sorted(set(rows) - set(source_ids))
    if missing or extra:
        raise ValueError(
            f"{split} source/audit mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    retained = [scenario for scenario in scenarios if is_pure_edge1(rows[scenario["scenario_id"]])]
    rejected = [scenario["scenario_id"] for scenario in scenarios if not is_pure_edge1(rows[scenario["scenario_id"]])]
    return retained, rejected


def build_payload(dataset_id, split, scenarios, config, sources):
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
    if args.monitor_per_pool < 1:
        raise ValueError("--monitor-per-pool must be positive")

    source_paths = {
        split: args.source / f"{split}.json.gz"
        for split in ("train", "validation")
    }
    source_payloads = {split: load_json(path) for split, path in source_paths.items()}
    audit = load_json(args.audit)
    if audit.get("protocol") != "manifest-full-path-conflict-horizon-audit-v1":
        raise ValueError(f"unsupported audit protocol: {audit.get('protocol')!r}")
    rows_by_split = audit_rows_by_split(audit)

    filtered = {}
    rejected = {}
    for split in source_paths:
        if split not in rows_by_split:
            raise ValueError(f"audit has no {split!r} rows")
        filtered[split], rejected[split] = filter_split(
            source_payloads[split], rows_by_split[split], split
        )

    monitor_pools = {}
    for pool in ("standard", "dense"):
        candidates = [
            scenario for scenario in filtered["validation"] if scenario["preset"] == pool
        ]
        random.Random(args.monitor_seed + (0 if pool == "standard" else 1)).shuffle(
            candidates
        )
        if len(candidates) < args.monitor_per_pool:
            raise ValueError(
                f"validation has only {len(candidates)} corrected {pool} scenarios"
            )
        monitor_pools[pool] = candidates[: args.monitor_per_pool]
    monitor = monitor_pools["standard"] + monitor_pools["dense"]
    random.Random(args.monitor_seed + 2).shuffle(monitor)

    sources = [
        {
            "split": split,
            "path": relative_path(path),
            "dataset_id": source_payloads[split].get("dataset_id"),
            "sha256": sha256(path),
            "scenario_count": len(source_payloads[split]["scenarios"]),
        }
        for split, path in source_paths.items()
    ]
    config = {
        "full_path_filter": {
            "conflict_edge_count": 1,
            "max_conflict_degree": 1,
            "simultaneous_conflict_count": 1,
        },
        "audit": {
            "path": relative_path(args.audit),
            "sha256": sha256(args.audit),
            "protocol": audit["protocol"],
            "parameters": audit.get("parameters", {}),
        },
        "rejected_scenario_ids": rejected,
        "policy_independent": True,
    }
    outputs = {
        "train": build_payload(
            "edge1-full-horizon-train-v1",
            "train",
            filtered["train"],
            config,
            sources,
        ),
        "validation": build_payload(
            "edge1-full-horizon-validation-v1",
            "validation",
            filtered["validation"],
            config,
            sources,
        ),
        "validation_monitor_50": build_payload(
            "edge1-full-horizon-monitor-50-v1",
            "validation",
            monitor,
            {
                **config,
                "monitor": {
                    "seed": args.monitor_seed,
                    "per_pool": args.monitor_per_pool,
                    "standard": len(monitor_pools["standard"]),
                    "dense": len(monitor_pools["dense"]),
                },
            },
            sources,
        ),
    }
    for name, payload in outputs.items():
        write_gzip_json(args.output / f"{name}.json.gz", payload)

    print(
        json.dumps(
            {
                "output": relative_path(args.output),
                "counts": {name: len(payload["scenarios"]) for name, payload in outputs.items()},
                "rejected": rejected,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
