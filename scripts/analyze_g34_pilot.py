#!/usr/bin/env python3
"""Audit and summarize the preregistered G34 Dense32 development pilot."""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/03_保留专门化/02_论文主线"
G34 = BASE / "34_双头RewardAwareGate"
MANIFEST = BASE / "datasets/fixed_v1/dense/validation.json.gz"
RESULTS = G34 / "local_data/pilot32/results"
FILES = {
    "g34": RESULTS / "g34_dense32_joint_s20260912.npy",
    "5a": RESULTS / "g34_dense32_5a_s20260912.npy",
}
OUTPUT = G34 / "local_data/pilot32/summary.json"
EXPECTED_MANIFEST_SHA256 = (
    "2d1dde389f927b924fa5993c47460bc60bac42aa9506ae3869c3139c9d1264b7"
)
EPISODES = 32
AGENTS = 5


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, expected_ids):
    rows = np.load(path, allow_pickle=True)
    if rows.shape != (EPISODES, 17):
        raise ValueError("invalid result shape for %s: %s" % (path, rows.shape))
    if [str(item) for item in rows[:, 12]] != expected_ids:
        raise ValueError("scenario order mismatch for %s" % path)
    terminal = sum(
        int(row[6]) + int(row[7]) + int(row[10]) for row in rows
    )
    if terminal != EPISODES * AGENTS:
        raise ValueError("terminal accounting mismatch for %s" % path)
    return rows


def aggregate(rows):
    return {
        "episodes": int(len(rows)),
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
        raise ValueError("Dense validation manifest hash mismatch")
    with gzip.open(MANIFEST, "rt", encoding="utf-8") as handle:
        expected_ids = [
            str(item["scenario_id"])
            for item in json.load(handle)["scenarios"][:EPISODES]
        ]
    rows = {name: load(path, expected_ids) for name, path in FILES.items()}
    metrics = {name: aggregate(value) for name, value in rows.items()}
    g34 = metrics["g34"]
    baseline = metrics["5a"]

    mode_diversity = (
        0.05 <= g34["interaction_selection_share"] <= 0.95
        and g34["total_switches"] > 0
    )
    full_success_non_worse = (
        g34["full_success_count"] >= baseline["full_success_count"]
    )
    collision_non_worse = (
        g34["robot_collision_count"] <= baseline["robot_collision_count"]
    )
    strictly_better_on_one = (
        g34["full_success_count"] > baseline["full_success_count"]
        or g34["robot_collision_count"] < baseline["robot_collision_count"]
    )
    if mode_diversity and full_success_non_worse and collision_non_worse and strictly_better_on_one:
        decision = "pass_for_64_episode_expansion"
    elif (
        mode_diversity
        and g34["full_success_count"] >= baseline["full_success_count"] - 1
        and g34["robot_collision_count"] <= baseline["robot_collision_count"] + 2
    ):
        decision = "inconclusive_stop_before_expansion"
    else:
        decision = "fail_stop_before_expansion"

    output = {
        "format_version": 1,
        "protocol": {
            "experiment_id": "G34-P1-Dense32-development",
            "episodes_per_method": EPISODES,
            "seed": 20260912,
            "manifest_slice": "dense validation first 32 scenes",
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
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
        "preregistered_checks": {
            "terminal_and_scenario_audit_passed": True,
            "both_actor_modes_selected": mode_diversity,
            "full_success_non_worse_than_same_seed_5a": full_success_non_worse,
            "collision_non_worse_than_same_seed_5a": collision_non_worse,
            "strictly_better_on_at_least_one_primary_outcome": strictly_better_on_one,
        },
        "decision": decision,
        "next_authorized_step": (
            "one fixed 64-episode development confirmation; no 256 run yet"
            if decision == "pass_for_64_episode_expansion"
            else "stop and diagnose without tuning on these 32 outcomes"
        ),
    }
    output["summary_sha256"] = hashlib.sha256(
        json.dumps(output, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
