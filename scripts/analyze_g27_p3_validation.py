#!/usr/bin/env python3
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/03_保留专门化/02_论文主线"
G27 = BASE / "27_反事实Reward增强Gate监督"
MANIFEST = BASE / "datasets/fixed_v1/dense/validation.json.gz"
REFERENCE = BASE / "12_参数匹配单Actor容量对照/local_data/dense_first256_pilot/results"
FILES = {
    "5a": BASE / "18_dense256当前方法复测/local_data/results/g18_dense256_5a_s20260810.npy",
    "b2": REFERENCE / "g12_dense256_b2_r1_s20260810.npy",
    "privileged_2m": REFERENCE / "g12_dense256_oracle_r1_s20260810.npy",
    "b3": G27 / "local_data/validation/results/g27_dense256_b3_s20260810.npy",
}
EXPECTED_HASHES = {
    "manifest": "2d1dde389f927b924fa5993c47460bc60bac42aa9506ae3869c3139c9d1264b7",
    "5a": "7300cd1175e7466baad66cdcf62bac431606202708c109752c2b6d40f592a2b3",
    "b2": "7dc2108b32759fec2dcde2b81c4547da9fb4f66e61e4202ad4807321e33bd86e",
    "privileged_2m": "6c7f942e1dfa61b9ce3f17217615ca406d72311e29e4eb0ce881d38c2e205c7c",
}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_result(path, expected_ids):
    rows = np.load(path, allow_pickle=True)
    if rows.shape != (256, 17):
        raise ValueError("invalid result shape for %s: %s" % (path, rows.shape))
    if [str(item) for item in rows[:, 12]] != expected_ids:
        raise ValueError("scenario order mismatch for %s" % path)
    if sum(int(row[6]) + int(row[7]) + int(row[10]) for row in rows) != 1280:
        raise ValueError("terminal accounting mismatch for %s" % path)
    return rows


def aggregate(rows):
    return {
        "episodes": len(rows),
        "full_success": float(rows[:, 8].astype(float).mean()),
        "agent_success": float(rows[:, 6].astype(float).sum() / (len(rows) * 5)),
        "robot_collision": float(rows[:, 7].astype(float).sum() / (len(rows) * 5)),
        "robot_unresolved": float(rows[:, 10].astype(float).sum() / (len(rows) * 5)),
        "episode_timeout": float(rows[:, 11].astype(float).mean()),
        "raw_steps": float(rows[:, 3].astype(float).mean()),
        "interaction_selection_share": float(rows[:, 14].astype(float).mean()),
        "switches": float(rows[:, 15].astype(float).mean()),
    }


def paired_effect(candidate, baseline):
    effects = {
        "full_success": candidate[:, 8].astype(float) - baseline[:, 8].astype(float),
        "robot_collision": (
            candidate[:, 7].astype(float) - baseline[:, 7].astype(float)
        )
        / 5.0,
        "episode_timeout": candidate[:, 11].astype(float)
        - baseline[:, 11].astype(float),
        "raw_steps": candidate[:, 3].astype(float) - baseline[:, 3].astype(float),
        "interaction_selection_share": candidate[:, 14].astype(float)
        - baseline[:, 14].astype(float),
        "switches": candidate[:, 15].astype(float) - baseline[:, 15].astype(float),
    }
    result = {
        name: {"mean_difference": float(values.mean())}
        for name, values in effects.items()
    }
    both_success = (candidate[:, 8].astype(int) == 1) & (
        baseline[:, 8].astype(int) == 1
    )
    paired_steps = (
        candidate[both_success, 3].astype(float)
        - baseline[both_success, 3].astype(float)
    )
    result["paired_success_steps"] = {
        "pairs": int(np.sum(both_success)),
        "mean_difference": float(np.mean(paired_steps)) if len(paired_steps) else None,
    }
    candidate_penalized = np.where(
        candidate[:, 8].astype(int) == 1, candidate[:, 3].astype(float), 300.0
    )
    baseline_penalized = np.where(
        baseline[:, 8].astype(int) == 1, baseline[:, 3].astype(float), 300.0
    )
    result["penalized_completion_steps"] = {
        "horizon": 300,
        "mean_difference": float(np.mean(candidate_penalized - baseline_penalized)),
    }
    return result


def main():
    if sha256_file(MANIFEST) != EXPECTED_HASHES["manifest"]:
        raise ValueError("Dense validation manifest hash mismatch")
    for method in ("5a", "b2", "privileged_2m"):
        if sha256_file(FILES[method]) != EXPECTED_HASHES[method]:
            raise ValueError("frozen %s validation result hash mismatch" % method)
    with gzip.open(MANIFEST, "rt", encoding="utf-8") as handle:
        expected_ids = [
            str(item["scenario_id"]) for item in json.load(handle)["scenarios"][:256]
        ]
    runs = {name: load_result(path, expected_ids) for name, path in FILES.items()}
    overall = {name: aggregate(rows) for name, rows in runs.items()}
    effect = paired_effect(runs["b3"], runs["b2"])
    paired_steps = effect["paired_success_steps"]["mean_difference"]
    criteria = {
        "full_success_not_below_b2": overall["b3"]["full_success"]
        >= overall["b2"]["full_success"],
        "robot_collision_increase_at_most_2pp": effect["robot_collision"]["mean_difference"]
        <= 0.02,
        "episode_timeout_increase_at_most_1pp": effect["episode_timeout"]["mean_difference"]
        <= 0.01,
        "interaction_share_down_10pp_or_paired_steps_down_5": (
            effect["interaction_selection_share"]["mean_difference"] <= -0.10
            or (paired_steps is not None and paired_steps <= -5.0)
        ),
    }
    checkpoint_summary = json.loads(
        (G27 / "local_data/training/seed20260910/summary.json").read_text(
            encoding="utf-8"
        )
    )
    output = {
        "format_version": 1,
        "protocol": {
            "experiment_id": "G27-P3-development-admission",
            "seed": 20260810,
            "episodes_per_method": 256,
            "manifest_sha256": EXPECTED_HASHES["manifest"],
            "reused_methods": ["5a", "b2", "privileged_2m"],
            "newly_run_methods": ["b3"],
            "sealed_test_read": False,
        },
        "result_sha256": {name: sha256_file(path) for name, path in FILES.items()},
        "b3_checkpoint_sha256": checkpoint_summary["checkpoint"]["sha256"],
        "overall": overall,
        "b3_minus_b2": effect,
        "admission_criteria": criteria,
        "admission_passed": all(criteria.values()),
        "next_action": (
            "freeze and run a new independent Dense test"
            if all(criteria.values())
            else "stop G27 and retain B2 as the paper method"
        ),
    }
    output["summary_sha256"] = hashlib.sha256(
        json.dumps(output, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()
    path = G27 / "local_data/validation/p3_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if not output["admission_passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
