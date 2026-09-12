#!/usr/bin/env python3
"""Audit and summarize the disjoint G34 Dense64 development confirmation."""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/03_保留专门化/02_论文主线"
G34 = BASE / "34_双头RewardAwareGate"
MANIFEST = G34 / "local_data/protocol/dense_confirmation64.json.gz"
RESULTS = G34 / "local_data/confirmation64/results"
FILES = {
    "g34": RESULTS / "g34_dense64_confirmation_joint_s20260913.npy",
    "5a": RESULTS / "g34_dense64_confirmation_5a_s20260913.npy",
}
OUTPUT = G34 / "local_data/confirmation64/summary.json"
EXPECTED_MANIFEST_SHA256 = (
    "4f70d1f7536f9619c5f687ea98c24d3a68b985e9a1d3673066d5df8c1f5ed4f7"
)
EPISODES = 64
AGENTS = 5


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, expected_ids):
    rows = np.load(path, allow_pickle=True)
    if rows.shape != (EPISODES, 17):
        raise ValueError("invalid result shape for %s: %s" % (path, rows.shape))
    if [str(item) for item in rows[:, 12]] != expected_ids:
        raise ValueError("scenario order mismatch for %s" % path)
    terminal = sum(int(row[6]) + int(row[7]) + int(row[10]) for row in rows)
    if terminal != EPISODES * AGENTS:
        raise ValueError("terminal accounting mismatch for %s" % path)
    return rows


def aggregate(rows):
    return {
        "episodes": EPISODES,
        "full_success_count": int(np.sum(rows[:, 8].astype(int))),
        "full_success_rate": float(np.mean(rows[:, 8].astype(float))),
        "agent_success_count": int(np.sum(rows[:, 6].astype(int))),
        "agent_success_rate": float(np.sum(rows[:, 6].astype(float)) / (EPISODES * AGENTS)),
        "robot_collision_count": int(np.sum(rows[:, 7].astype(int))),
        "robot_collision_rate": float(np.sum(rows[:, 7].astype(float)) / (EPISODES * AGENTS)),
        "robot_unresolved_count": int(np.sum(rows[:, 10].astype(int))),
        "episode_timeout_count": int(np.sum(rows[:, 11].astype(int))),
        "mean_environment_steps": float(np.mean(rows[:, 3].astype(float))),
        "interaction_selection_share": float(np.mean(rows[:, 14].astype(float))),
        "total_switches": int(np.sum(rows[:, 15].astype(int))),
    }


def main():
    if sha256(MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("G34 Dense64 manifest hash mismatch")
    with gzip.open(MANIFEST, "rt", encoding="utf-8") as handle:
        expected_ids = [str(item["scenario_id"]) for item in json.load(handle)["scenarios"]]
    rows = {name: load(path, expected_ids) for name, path in FILES.items()}
    metrics = {name: aggregate(value) for name, value in rows.items()}
    g34 = metrics["g34"]
    baseline = metrics["5a"]

    checks = {
        "terminal_and_scenario_audit_passed": True,
        "both_actor_modes_selected": (
            0.05 <= g34["interaction_selection_share"] <= 0.95
            and g34["total_switches"] > 0
        ),
        "full_success_non_worse_than_same_seed_5a": (
            g34["full_success_count"] >= baseline["full_success_count"]
        ),
        "collision_non_worse_than_same_seed_5a": (
            g34["robot_collision_count"] <= baseline["robot_collision_count"]
        ),
        "strictly_better_on_at_least_one_primary_outcome": (
            g34["full_success_count"] > baseline["full_success_count"]
            or g34["robot_collision_count"] < baseline["robot_collision_count"]
        ),
    }
    passed = all(checks.values())
    output = {
        "format_version": 1,
        "protocol": {
            "experiment_id": "G34-P2-Dense64-development-confirmation",
            "episodes_per_method": EPISODES,
            "seed": 20260913,
            "manifest_source_slice": "dense validation [32:96]",
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "overlap_with_dense32_pilot": 0,
            "methods": ["G34 joint-supervision Router", "frozen 5A"],
            "b2_used_as_adoption_baseline": False,
            "inferential_claim_authorized": False,
        },
        "result_sha256": {name: sha256(path) for name, path in FILES.items()},
        "overall": metrics,
        "g34_minus_5a": {
            "full_success_count": g34["full_success_count"] - baseline["full_success_count"],
            "full_success_rate": g34["full_success_rate"] - baseline["full_success_rate"],
            "robot_collision_count": g34["robot_collision_count"] - baseline["robot_collision_count"],
            "robot_collision_rate": g34["robot_collision_rate"] - baseline["robot_collision_rate"],
        },
        "preregistered_checks": checks,
        "decision": (
            "pass_for_independent_evaluation_design"
            if passed
            else "fail_stop_before_independent_evaluation"
        ),
        "next_authorized_step": (
            "freeze G34 and register an independent evaluation; do not launch it automatically"
            if passed
            else "stop and diagnose without tuning on these outcomes"
        ),
    }
    output["summary_sha256"] = hashlib.sha256(
        json.dumps(output, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
