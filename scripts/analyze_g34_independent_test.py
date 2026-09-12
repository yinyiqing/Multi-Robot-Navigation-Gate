#!/usr/bin/env python3
"""Analyze the frozen G34 independent Dense test with scene-cluster inference."""

import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_g27_p4_test import bca_mean_interval, sign_flip_p


BASE = ROOT / "experiments/03_保留专门化/02_论文主线"
G34 = BASE / "34_双头RewardAwareGate"
TEST = G34 / "local_data/independent_test"
PROTOCOL = TEST / "protocol.json"
MANIFEST = TEST / "manifest/dense_test_640_896.json.gz"
RECORD = TEST / "manifest/record.json"
COMPLETION = TEST / "completion.json"
RESULT_DIR = TEST / "results"
OUTPUT = TEST / "statistics.json"
CHECKPOINT = G34 / "local_data/training/seed20260912/best_runtime.pt"
METHODS = ("5a", "g34")
SEEDS = (20260914, 20260915, 20260916)
SCENES = 256
AGENTS = 5
HORIZON = 300
BOOTSTRAP_SEED = 20260912


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def aggregate(rows):
    return {
        "episodes": int(len(rows)),
        "full_success": float(rows[:, 8].astype(float).mean()),
        "agent_success": float(rows[:, 6].astype(float).sum() / (len(rows) * AGENTS)),
        "robot_collision": float(rows[:, 7].astype(float).sum() / (len(rows) * AGENTS)),
        "robot_unresolved": float(rows[:, 10].astype(float).sum() / (len(rows) * AGENTS)),
        "episode_timeout": float(rows[:, 11].astype(float).mean()),
        "environment_steps": float(rows[:, 3].astype(float).mean()),
        "interaction_selection_share": float(rows[:, 14].astype(float).mean()),
        "switches_per_episode": float(rows[:, 15].astype(float).mean()),
    }


def metric_matrix(runs, method, column, scale=1.0):
    return np.stack(
        [runs[(method, seed)][:, column].astype(float) / scale for seed in SEEDS],
        axis=1,
    )


def effect(values, sign_flip=False):
    values = np.asarray(values, dtype=float)
    result = {
        "mean_difference": float(values.mean()),
        "scene_cluster_bca_95_ci": bca_mean_interval(
            values, samples=20000, seed=BOOTSTRAP_SEED
        ),
    }
    if sign_flip:
        result["sign_flip_two_sided_p"] = sign_flip_p(
            values, samples=100000, seed=BOOTSTRAP_SEED
        )
    return result


def main():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    completion = json.loads(COMPLETION.read_text(encoding="utf-8"))
    if protocol.get("experiment_id") != "G34-P3-independent-dense-test":
        raise ValueError("invalid G34 independent protocol")
    if protocol["method"].get("checkpoint_sha256") != sha256_file(CHECKPOINT):
        raise ValueError("G34 checkpoint changed after registration")
    if record.get("protocol_sha256") != sha256_file(PROTOCOL):
        raise ValueError("manifest record does not match protocol")
    if record.get("output_sha256") != sha256_file(MANIFEST):
        raise ValueError("G34 independent manifest hash mismatch")
    if any(
        values.get("scene_id_overlap") or values.get("complete_geometry_overlap")
        for values in record.get("overlap_audit", {}).values()
    ):
        raise ValueError("G34 independent overlap audit is not clean")
    if completion.get("status") != "complete":
        raise ValueError("G34 independent test is incomplete")
    if completion.get("manifest_sha256") != sha256_file(MANIFEST):
        raise ValueError("completion manifest hash mismatch")

    with gzip.open(MANIFEST, "rt", encoding="utf-8") as handle:
        expected_ids = [str(row["scenario_id"]) for row in json.load(handle)["scenarios"]]
    if len(expected_ids) != SCENES or len(set(expected_ids)) != SCENES:
        raise ValueError("independent manifest must have 256 unique scenes")

    runs = {}
    result_hashes = {}
    for method in METHODS:
        for seed in SEEDS:
            key = "%s_s%d" % (method, seed)
            path = RESULT_DIR / ("g34_p3_%s_s%d.npy" % (method, seed))
            rows = np.load(path, allow_pickle=True)
            if rows.shape != (SCENES, 17):
                raise ValueError("invalid result shape: %s" % path)
            if [str(item) for item in rows[:, 12]] != expected_ids:
                raise ValueError("scenario order mismatch: %s" % path)
            if sum(int(row[6]) + int(row[7]) + int(row[10]) for row in rows) != SCENES * AGENTS:
                raise ValueError("terminal accounting mismatch: %s" % path)
            result_hashes[key] = sha256_file(path)
            if completion.get("result_sha256", {}).get(key) != result_hashes[key]:
                raise ValueError("completion result hash mismatch: %s" % path)
            runs[(method, seed)] = rows

    overall = {
        method: aggregate(np.concatenate([runs[(method, seed)] for seed in SEEDS], axis=0))
        for method in METHODS
    }
    metric_specs = {
        "full_success": (8, 1.0),
        "agent_success": (6, float(AGENTS)),
        "robot_collision": (7, float(AGENTS)),
        "robot_unresolved": (10, float(AGENTS)),
        "episode_timeout": (11, 1.0),
        "environment_steps": (3, 1.0),
        "interaction_selection_share": (14, 1.0),
        "switches": (15, 1.0),
    }
    effects = {}
    for name, (column, scale) in metric_specs.items():
        baseline = metric_matrix(runs, "5a", column, scale)
        candidate = metric_matrix(runs, "g34", column, scale)
        effects[name] = effect(
            (candidate - baseline).mean(axis=1), sign_flip=(name == "full_success")
        )

    baseline_full = metric_matrix(runs, "5a", 8)
    candidate_full = metric_matrix(runs, "g34", 8)
    baseline_steps = metric_matrix(runs, "5a", 3)
    candidate_steps = metric_matrix(runs, "g34", 3)
    paired_steps = []
    joint_success = (baseline_full == 1) & (candidate_full == 1)
    for scene in range(SCENES):
        mask = joint_success[scene]
        if np.any(mask):
            paired_steps.append(float(np.mean((candidate_steps - baseline_steps)[scene, mask])))
    effects["paired_success_steps"] = {
        "joint_success_scene_repeat_pairs": int(np.sum(joint_success)),
        "scene_clusters_with_pairs": len(paired_steps),
        **effect(paired_steps),
    }
    baseline_penalized = np.where(baseline_full == 1, baseline_steps, HORIZON)
    candidate_penalized = np.where(candidate_full == 1, candidate_steps, HORIZON)
    effects["penalized_completion_steps"] = {
        "horizon": HORIZON,
        **effect((candidate_penalized - baseline_penalized).mean(axis=1)),
    }

    full = effects["full_success"]
    collision = effects["robot_collision"]
    claim_checks = {
        "full_success_positive": full["mean_difference"] > 0.0,
        "full_success_bca_lower_above_zero": full["scene_cluster_bca_95_ci"][0] > 0.0,
        "full_success_sign_flip_p_below_0_05": full["sign_flip_two_sided_p"] < 0.05,
        "collision_difference_negative": collision["mean_difference"] < 0.0,
        "collision_bca_upper_below_zero": collision["scene_cluster_bca_95_ci"][1] < 0.0,
        "both_actor_modes_selected": (
            0.05 <= overall["g34"]["interaction_selection_share"] <= 0.95
            and overall["g34"]["switches_per_episode"] > 0.0
        ),
    }
    output = {
        "format_version": 1,
        "experiment_id": "G34-P3-independent-dense-test",
        "protocol_sha256": sha256_file(PROTOCOL),
        "manifest_sha256": sha256_file(MANIFEST),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "result_sha256": result_hashes,
        "overall": overall,
        "g34_minus_5a": effects,
        "claim_checks": claim_checks,
        "primary_and_safety_claim_supported": all(claim_checks.values()),
        "interpretation_boundary": (
            "Independent evidence for G34 over frozen 5A; no comparison with B2 and no post-result tuning."
        ),
    }
    output["statistics_sha256"] = canonical_sha256(output)
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
