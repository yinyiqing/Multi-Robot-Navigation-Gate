#!/usr/bin/env python3
import gzip
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/03_保留专门化/02_论文主线"
RUN_DIR = BASE / "18_dense256当前方法复测/local_data"
MANIFEST = BASE / "datasets/fixed_v1/dense/validation.json.gz"
REFERENCE_DIR = BASE / "12_参数匹配单Actor容量对照/local_data/dense_first256_pilot/results"
SEED = 20260810

FILES = {
    "5a": REFERENCE_DIR / "g12_dense256_5a_r1_s20260810.npy",
    "b2": REFERENCE_DIR / "g12_dense256_b2_r1_s20260810.npy",
    "oracle_2m": REFERENCE_DIR / "g12_dense256_oracle_r1_s20260810.npy",
    "r2_10k": REFERENCE_DIR / "g12_dense256_r2_10k_r1_s20260810.npy",
    "r2b_best": RUN_DIR / "results/g18_dense256_r2b_best_s20260810.npy",
}


def edge_band(value):
    value = int(value)
    return str(value) if value < 3 else "3+"


def load(path, expected_ids):
    rows = np.load(path, allow_pickle=True)
    if rows.shape != (256, 17):
        raise ValueError(f"invalid result shape for {path}: {rows.shape}")
    if [str(row[12]) for row in rows] != expected_ids:
        raise ValueError(f"scenario order mismatch for {path}")
    if sum(int(row[6]) + int(row[7]) + int(row[10]) for row in rows) != 1280:
        raise ValueError(f"terminal accounting mismatch for {path}")
    return rows


def aggregate(rows):
    count = len(rows)
    return {
        "episodes": count,
        "full_success": float(rows[:, 8].astype(int).mean()),
        "agent_success": float(rows[:, 6].astype(int).sum() / (count * 5)),
        "collision": float(rows[:, 7].astype(int).sum() / (count * 5)),
        "unresolved": float(rows[:, 10].astype(int).sum() / (count * 5)),
        "timeout": float(rows[:, 11].astype(int).mean()),
        "mean_steps": float(rows[:, 3].astype(float).mean()),
    }


def main():
    with gzip.open(MANIFEST, "rt", encoding="utf-8") as handle:
        scenarios = json.load(handle)["scenarios"][:256]
    expected_ids = [str(item["scenario_id"]) for item in scenarios]
    bands = {
        str(item["scenario_id"]): edge_band(item["metrics"]["conflict_edge_count"])
        for item in scenarios
    }
    runs = {name: load(path, expected_ids) for name, path in FILES.items()}
    summary = {
        "protocol": {
            "experiment_id": "G18-dense256-r2b",
            "seed": SEED,
            "episodes_per_method": 256,
            "sealed_test_read": False,
        },
        "overall": {name: aggregate(rows) for name, rows in runs.items()},
        "by_conflict_edges": {},
    }
    for band in ("0", "1", "2", "3+"):
        summary["by_conflict_edges"][band] = {}
        for name, rows in runs.items():
            mask = np.asarray([bands[str(row[12])] == band for row in rows])
            summary["by_conflict_edges"][band][name] = aggregate(rows[mask])
    output = RUN_DIR / "summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
