#!/usr/bin/env python3
"""Summarize the single G28 B4 Dense256 development rollout."""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/03_保留专门化/02_论文主线"
G28 = BASE / "28_RewardAwareGate稳健修正版"
MANIFEST = BASE / "datasets/fixed_v1/dense/validation.json.gz"
FILES = {
    "5a": BASE / "18_dense256当前方法复测/local_data/results/g18_dense256_5a_s20260810.npy",
    "b2": BASE / "12_参数匹配单Actor容量对照/local_data/dense_first256_pilot/results/g12_dense256_b2_r1_s20260810.npy",
    "b4": G28 / "local_data/validation/results/g28_dense256_b4_s20260911.npy",
}
SAME_SEED_B2 = G28 / "local_data/validation/results/g28_dense256_b2_s20260911.npy"
EXPECTED = {
    "manifest": "2d1dde389f927b924fa5993c47460bc60bac42aa9506ae3869c3139c9d1264b7",
    "5a": "7300cd1175e7466baad66cdcf62bac431606202708c109752c2b6d40f592a2b3",
    "b2": "7dc2108b32759fec2dcde2b81c4547da9fb4f66e61e4202ad4807321e33bd86e",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, expected_ids):
    rows = np.load(path, allow_pickle=True)
    if rows.shape != (256, 17):
        raise ValueError("invalid shape for %s: %s" % (path, rows.shape))
    if [str(item) for item in rows[:, 12]] != expected_ids:
        raise ValueError("scenario order mismatch for %s" % path)
    if sum(int(row[6]) + int(row[7]) + int(row[10]) for row in rows) != 1280:
        raise ValueError("terminal accounting mismatch for %s" % path)
    return rows


def aggregate(rows):
    return {
        "episodes": len(rows),
        "full_success": float(np.mean(rows[:, 8].astype(float))),
        "agent_success": float(np.sum(rows[:, 6].astype(float)) / 1280),
        "robot_collision": float(np.sum(rows[:, 7].astype(float)) / 1280),
        "robot_unresolved": float(np.sum(rows[:, 10].astype(float)) / 1280),
        "episode_timeout": float(np.mean(rows[:, 11].astype(float))),
        "raw_steps": float(np.mean(rows[:, 3].astype(float))),
        "interaction_selection_share": float(np.mean(rows[:, 14].astype(float))),
        "switches": float(np.mean(rows[:, 15].astype(float))),
    }


def effect(candidate, baseline):
    values = {
        "full_success": candidate[:, 8].astype(float) - baseline[:, 8].astype(float),
        "robot_collision": (candidate[:, 7].astype(float) - baseline[:, 7].astype(float)) / 5.0,
        "episode_timeout": candidate[:, 11].astype(float) - baseline[:, 11].astype(float),
        "raw_steps": candidate[:, 3].astype(float) - baseline[:, 3].astype(float),
        "interaction_selection_share": candidate[:, 14].astype(float) - baseline[:, 14].astype(float),
        "switches": candidate[:, 15].astype(float) - baseline[:, 15].astype(float),
    }
    out = {name: {"mean_difference": float(np.mean(value))} for name, value in values.items()}
    both = (candidate[:, 8].astype(int) == 1) & (baseline[:, 8].astype(int) == 1)
    out["paired_success_steps"] = {
        "pairs": int(np.sum(both)),
        "mean_difference": float(np.mean(candidate[both, 3].astype(float) - baseline[both, 3].astype(float))) if np.any(both) else None,
    }
    return out


def main():
    if sha256(MANIFEST) != EXPECTED["manifest"]:
        raise ValueError("manifest hash mismatch")
    for name in ("5a", "b2"):
        if sha256(FILES[name]) != EXPECTED[name]:
            raise ValueError("frozen %s result hash mismatch" % name)
    with gzip.open(MANIFEST, "rt", encoding="utf-8") as handle:
        ids = [str(item["scenario_id"]) for item in json.load(handle)["scenarios"][:256]]
    runs = {name: load(path, ids) for name, path in FILES.items()}
    same_seed_available = False
    if SAME_SEED_B2.is_file():
        try:
            same_seed_available = np.load(SAME_SEED_B2, allow_pickle=True).shape == (256, 17)
        except Exception:
            same_seed_available = False
    if same_seed_available:
        runs["b2_same_seed"] = load(SAME_SEED_B2, ids)
    output = {
        "format_version": 1,
        "protocol": {
            "experiment_id": "G28-P3-development",
            "b4_seed": 20260911,
            "baseline_seed": 20260810,
            "episodes": 256,
            "pairing": "same manifest, different seeds; no paired inferential test",
            "sealed_test_read": False,
        },
        "result_sha256": {name: sha256(path) for name, path in FILES.items()},
        "overall": {name: aggregate(rows) for name, rows in runs.items()},
        "b4_minus_b2_unpaired_seed_direction": effect(runs["b4"], runs["b2"]),
        "b4_minus_5a_unpaired_seed_direction": effect(runs["b4"], runs["5a"]),
    }
    if same_seed_available:
        output["result_sha256"]["b2_same_seed"] = sha256(SAME_SEED_B2)
        output["b4_minus_b2_same_seed"] = effect(runs["b4"], runs["b2_same_seed"])
    output["summary_sha256"] = hashlib.sha256(json.dumps(output, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = G28 / "local_data/validation/p3_summary.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
