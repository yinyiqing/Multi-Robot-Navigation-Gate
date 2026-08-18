#!/usr/bin/env python3
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/03_保留专门化/02_论文主线"
G25 = BASE / "25_最终消融与Sealed评测/local_data"
G12_REFERENCE = BASE / "12_参数匹配单Actor容量对照/local_data/dense_first256_pilot/results"

FILES = {
    "5a": BASE / "18_dense256当前方法复测/local_data/results/g18_dense256_5a_s20260810.npy",
    "b2": G12_REFERENCE / "g12_dense256_b2_r1_s20260810.npy",
    "v1_epoch16_always_on": G12_REFERENCE / "g12_dense256_epoch16_r1_s20260810.npy",
    "v2_min_lidar": G25 / "results/g25_dense256_min_lidar_s20260810.npy",
    "v3_ttc_cpa": G25 / "results/g25_dense256_v3_ttc_cpa_s20260810.npy",
    "v4_single_frame": G25 / "results/g25_dense256_v4_single_frame_s20260810.npy",
    "v5_no_action_difference": G25 / "results/g25_dense256_v5_no_action_difference_s20260810.npy",
    "v6_no_hysteresis_hold": G25 / "results/g25_dense256_b2_no_hysteresis_hold_s20260810.npy",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate(rows):
    episodes = len(rows)
    agents = episodes * 5
    return {
        "episodes": episodes,
        "full_success": float(rows[:, 8].astype(int).mean()),
        "agent_success": float(rows[:, 6].astype(int).sum() / agents),
        "collision": float(rows[:, 7].astype(int).sum() / agents),
        "unresolved": float(rows[:, 10].astype(int).sum() / agents),
        "timeout": float(rows[:, 11].astype(int).mean()),
        "raw_mean_steps": float(rows[:, 3].astype(float).mean()),
        "interaction_share": float(rows[:, 14].astype(float).mean()),
        "mean_switches": float(rows[:, 15].astype(float).mean()),
    }


def exact_p(improved, degraded):
    count = improved + degraded
    if count == 0:
        return 1.0
    lower = min(improved, degraded)
    tail = sum(math.comb(count, index) for index in range(lower + 1))
    return min(1.0, 2.0 * tail / (2**count))


def paired(candidate, baseline):
    candidate_full = candidate[:, 8].astype(int)
    baseline_full = baseline[:, 8].astype(int)
    improved = int(np.sum(candidate_full > baseline_full))
    degraded = int(np.sum(candidate_full < baseline_full))
    return {
        "improved": improved,
        "degraded": degraded,
        "tied": len(candidate) - improved - degraded,
        "mcnemar_exact_two_sided_p": exact_p(improved, degraded),
    }


def main():
    manifest = BASE / "datasets/fixed_v1/dense/validation.json.gz"
    with gzip.open(manifest, "rt", encoding="utf-8") as handle:
        scenarios = json.load(handle)["scenarios"][:256]
    expected_ids = [str(item["scenario_id"]) for item in scenarios]
    runs = {}
    hashes = {}
    for name, path in FILES.items():
        rows = np.load(path, allow_pickle=True)
        if rows.shape != (256, 17):
            raise ValueError("invalid result shape for %s: %s" % (name, rows.shape))
        if [str(item) for item in rows[:, 12]] != expected_ids:
            raise ValueError("scenario order mismatch for %s" % name)
        terminal_count = sum(
            int(row[6]) + int(row[7]) + int(row[10]) for row in rows
        )
        if terminal_count != 1280:
            raise ValueError("terminal accounting mismatch for %s" % name)
        runs[name] = rows
        hashes[name] = sha256_file(path)

    summary = {
        "protocol": {
            "experiment_id": "G25-validation-ablations",
            "seed": 20260810,
            "episodes_per_method": 256,
            "manifest": str(manifest.relative_to(ROOT)),
            "manifest_sha256": sha256_file(manifest),
            "sealed_test_read": False,
        },
        "result_sha256": hashes,
        "overall": {name: aggregate(rows) for name, rows in runs.items()},
        "paired_vs_5a": {
            name: paired(rows, runs["5a"])
            for name, rows in runs.items()
            if name != "5a"
        },
        "paired_vs_b2": {
            name: paired(rows, runs["b2"])
            for name, rows in runs.items()
            if name != "b2"
        },
    }
    output = G25 / "validation_ablations_summary.json"
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
