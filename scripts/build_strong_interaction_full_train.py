#!/usr/bin/env python3
"""Build the fixed, de-duplicated strong-interaction training pool.

Only source train manifests are read. Validation and test manifests are never
included in the output.
"""
import argparse
import gzip
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "TD3"))
from scenario_manifests import AGENT_NAMES, load_manifest_dataset, validate_manifest_scenarios


ROOT = Path(__file__).resolve().parents[1]
BANDS = (("deep", 0.0, 0.4), ("close", 0.4, 0.6), ("margin", 0.6, 0.9))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def band(scenario):
    metrics = scenario.get("metrics", {})
    if metrics.get("conflict_edge_count") != 1:
        return None
    separation = metrics.get("min_synchronized_path_separation_m")
    if separation is None:
        return None
    for name, low, high in BANDS:
        if low <= float(separation) < high:
            return name
    return None


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()
    with path.open("wb") as f, gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as gz:
        gz.write(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=ROOT / "experiments/04_保留专门化/05_论文主线/datasets/fixed_v1")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260726)
    args = ap.parse_args()

    scenarios = []
    sources = []
    seen = set()
    for pool in ("standard", "dense"):
        path = args.dataset_root / pool / "train.json.gz"
        data = load_manifest_dataset(path)
        valid = validate_manifest_scenarios(data["scenarios"], AGENT_NAMES)
        sources.append({"pool": pool, "split": "train", "path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
        for item in valid:
            scenario = dict(item)
            scenario_band = band(scenario)
            if scenario_band is None:
                continue
            sid = scenario.get("scenario_id")
            if not sid or sid in seen:
                continue
            seen.add(sid)
            scenario["view"] = {"interaction_band": scenario_band, "purpose": "strong_interaction_full_train", "policy_independent": True}
            scenarios.append(scenario)

    random.Random(args.seed).shuffle(scenarios)
    counts = {name: sum(s["view"]["interaction_band"] == name for s in scenarios) for name, _, _ in BANDS}
    payload = {
        "dataset_version": 1,
        "dataset_id": "strong-interaction-full-train-v1",
        "split": "train",
        "view_config": {"purpose": "strong_interaction_full_train", "risk_bands": {n: {"min_inclusive_m": lo, "max_exclusive_m": hi} for n, lo, hi in BANDS}, "policy_independent": True, "seed": args.seed},
        "source_manifests": sources,
        "scenarios": scenarios,
    }
    write(args.output, payload)
    print(json.dumps({"scenarios": len(scenarios), "bands": counts, "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
