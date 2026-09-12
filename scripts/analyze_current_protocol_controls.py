#!/usr/bin/env python3
"""Summarize current-protocol control runs without importing historical rows."""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/03_保留专门化/02_论文主线"
EXP = BASE / "35_当前协议对照补测"
G34 = BASE / "34_双头RewardAwareGate/local_data/independent_test"
MANIFEST = G34 / "manifest/dense_test_640_896.json.gz"
RESULT_DIR = EXP / "local_data/results"
COMPLETION = EXP / "local_data/completion.json"
OUTPUT = EXP / "local_data/statistics.json"
AGENTS = 5

import sys
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_g27_p4_test import bca_mean_interval, sign_flip_p


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate(rows):
    n = len(rows)
    return {
        "episodes": int(n),
        "full_success_percent": float(rows[:, 8].astype(float).mean() * 100),
        "agent_success_percent": float(rows[:, 6].astype(float).sum() / (n * AGENTS) * 100),
        "robot_collision_percent": float(rows[:, 7].astype(float).sum() / (n * AGENTS) * 100),
        "robot_unresolved_percent": float(rows[:, 10].astype(float).sum() / (n * AGENTS) * 100),
        "timeout_percent": float(rows[:, 11].astype(float).mean() * 100),
        "mean_environment_steps": float(rows[:, 3].astype(float).mean()),
        "interaction_selection_share_percent": float(rows[:, 14].astype(float).mean() * 100),
        "mean_switches": float(rows[:, 15].astype(float).mean()),
    }


def main():
    completion = json.loads(COMPLETION.read_text(encoding="utf-8"))
    episodes = int(completion["episodes_per_method_repeat"])
    seeds = [int(x) for x in completion["seeds"]]
    methods = list(completion["methods"])
    with gzip.open(MANIFEST, "rt", encoding="utf-8") as handle:
        expected = [str(x["scenario_id"]) for x in json.load(handle)["scenarios"][:episodes]]

    runs = {}
    hashes = {}
    for method in methods:
        for seed in seeds:
            path = RESULT_DIR / f"controls_{method}_s{seed}_n{episodes}.npy"
            rows = np.load(path, allow_pickle=True)
            if rows.shape != (episodes, 17):
                raise ValueError(f"wrong shape: {path}")
            if [str(x) for x in rows[:, 12]] != expected:
                raise ValueError(f"scenario order mismatch: {path}")
            if sum(int(r[6]) + int(r[7]) + int(r[10]) for r in rows) != episodes * AGENTS:
                raise ValueError(f"terminal accounting mismatch: {path}")
            runs[(method, seed)] = rows
            hashes[f"{method}_s{seed}"] = sha256(path)

    baseline = {}
    for seed in seeds:
        path = G34 / f"results/g34_p3_5a_s{seed}.npy"
        if not path.is_file():
            raise ValueError(f"missing G34 Navigation Actor row for seed {seed}")
        rows = np.load(path, allow_pickle=True)
        if rows.shape[0] < episodes or [str(x) for x in rows[:episodes, 12]] != expected:
            raise ValueError(f"G34 baseline is not aligned for seed {seed}")
        baseline[seed] = rows[:episodes]

    overall = {method: aggregate(np.concatenate([runs[(method, seed)] for seed in seeds])) for method in methods}
    overall["navigation_actor"] = aggregate(np.concatenate([baseline[seed] for seed in seeds]))
    effects = {}
    for method in methods:
        candidate = np.stack([runs[(method, seed)][:, 8].astype(float) for seed in seeds], axis=1)
        base = np.stack([baseline[seed][:, 8].astype(float) for seed in seeds], axis=1)
        diff = (candidate - base).mean(axis=1)
        effects[method] = {
            "full_success_difference_percent": float(diff.mean() * 100),
            "scene_cluster_bca_95_ci_percent": [float(x * 100) for x in bca_mean_interval(diff, samples=20000, seed=20260912)],
            "two_sided_sign_flip_p": float(sign_flip_p(diff, samples=100000, seed=20260912)),
        }

    output = {
        "format_version": 1,
        "experiment_id": "current-protocol-controls",
        "manifest_sha256": sha256(MANIFEST),
        "episodes_per_method_repeat": episodes,
        "seeds": seeds,
        "methods": methods,
        "result_sha256": hashes,
        "overall": overall,
        "vs_navigation_actor": effects,
        "interpretation": "All control rows use the current independent-test slice; no G25/G26 values are imported.",
    }
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
