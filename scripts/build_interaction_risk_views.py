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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build policy-independent geometric risk views for edge-1 scenarios."
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
    parser.add_argument("--probe-per-pool", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def risk_band(scenario):
    metrics = scenario.get("metrics", {})
    if metrics.get("conflict_edge_count") != 1:
        return None
    separation = float(metrics["min_synchronized_path_separation_m"])
    for name, lower, upper in RISK_BANDS:
        if lower <= separation < upper:
            return name
    return None


def write_gzip_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as gzip_handle:
            gzip_handle.write(encoded)


def make_payload(dataset_id, split, scenarios, source_records, view_config):
    return {
        "dataset_version": 1,
        "dataset_id": dataset_id,
        "split": split,
        "view_config": view_config,
        "source_manifests": source_records,
        "scenarios": scenarios,
    }


def main():
    args = parse_args()
    if args.probe_per_pool < 1:
        raise ValueError("--probe-per-pool must be positive")

    pools = ("standard", "dense")
    splits = ("train", "validation")
    grouped = {
        band: {pool: {split: [] for split in splits} for pool in pools}
        for band, _, _ in RISK_BANDS
    }
    source_records = []

    for pool in pools:
        for split in splits:
            path = args.dataset_root / pool / f"{split}.json.gz"
            payload = load_manifest_dataset(path)
            scenarios = validate_manifest_scenarios(payload["scenarios"], AGENT_NAMES)
            for scenario in scenarios:
                band = risk_band(scenario)
                if band is not None:
                    grouped[band][pool][split].append(scenario)
            source_records.append(
                {
                    "pool": pool,
                    "split": split,
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "dataset_id": payload.get("dataset_id"),
                    "sha256": sha256(path),
                }
            )

    thresholds = {
        name: {"min_inclusive_m": lower, "max_exclusive_m": upper}
        for name, lower, upper in RISK_BANDS
    }
    common_config = {
        "conflict_edge_count": 1,
        "risk_metric": "min_synchronized_path_separation_m",
        "risk_bands": thresholds,
        "policy_independent": True,
        "shuffle_seed": args.seed,
    }
    summary = {"risk_bands": thresholds, "bands": {}}
    combined_probe = []

    for band_index, (band, _, _) in enumerate(RISK_BANDS):
        band_dir = args.output / band
        train = []
        validation = []
        band_counts = {}
        for pool_index, pool in enumerate(pools):
            pool_train = list(grouped[band][pool]["train"])
            pool_validation = list(grouped[band][pool]["validation"])
            random.Random(args.seed + band_index * 100 + pool_index).shuffle(pool_train)
            if len(pool_train) < args.probe_per_pool:
                raise ValueError(
                    f"{band}/{pool} has only {len(pool_train)} train scenarios, "
                    f"need {args.probe_per_pool}"
                )
            train.extend(pool_train[: args.probe_per_pool])
            validation.extend(pool_validation)
            band_counts[pool] = {
                "train_available": len(pool_train),
                "probe": args.probe_per_pool,
                "validation": len(pool_validation),
            }
        random.Random(args.seed + band_index * 1000 + 500).shuffle(train)
        random.Random(args.seed + band_index * 1000 + 600).shuffle(validation)
        combined_probe.extend(train)

        config = dict(common_config)
        config.update(
            {
                "selected_risk_band": band,
                "probe_per_pool": args.probe_per_pool,
            }
        )
        probe_path = band_dir / "probe.json.gz"
        validation_path = band_dir / "validation.json.gz"
        write_gzip_json(
            probe_path,
            make_payload(
                f"interaction-risk-{band}-probe-v1",
                "train",
                train,
                source_records,
                config,
            ),
        )
        write_gzip_json(
            validation_path,
            make_payload(
                f"interaction-risk-{band}-validation-v1",
                "validation",
                validation,
                source_records,
                config,
            ),
        )
        summary["bands"][band] = {
            "counts": band_counts,
            "probe_path": str(probe_path),
            "probe_scenarios": len(train),
            "validation_path": str(validation_path),
            "validation_scenarios": len(validation),
        }

    random.Random(args.seed + 5000).shuffle(combined_probe)
    combined_path = args.output / "probe.json.gz"
    combined_config = dict(common_config)
    combined_config.update(
        {
            "selected_risk_band": "all",
            "probe_per_pool_per_band": args.probe_per_pool,
        }
    )
    write_gzip_json(
        combined_path,
        make_payload(
            "interaction-risk-balanced-probe-v1",
            "train",
            combined_probe,
            source_records,
            combined_config,
        ),
    )
    summary["combined_probe"] = {
        "path": str(combined_path),
        "scenarios": len(combined_probe),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
