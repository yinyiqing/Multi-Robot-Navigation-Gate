#!/usr/bin/env python3
import gzip
import json
import math
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = (
    ROOT
    / "experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究"
    / "G11_F_epoch17_gate_v1"
)
MANIFEST = (
    ROOT
    / "experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views"
    / "g11_c_pilot_v1/validation.json.gz"
)
RESULT_DIR = RUN_DIR / "local_data/pilot/results"
ACTIVE_LOG_DIR = ROOT / "logs/active/g11_f_epoch17_gate_r2_pilot"
ARCHIVE_LOG_DIR = ROOT / "logs/archive/validation/g11_f_epoch17_gate_r2_pilot"
POLICIES = ("5a", "a1", "b2", "r2")
REPEATS = {1: 20260805, 2: 20260806}


def aggregate(rows):
    episodes = len(rows)
    success = sum(int(row[6]) for row in rows)
    collision = sum(int(row[7]) for row in rows)
    unresolved = sum(int(row[10]) for row in rows)
    if success + collision + unresolved != episodes * 5:
        raise ValueError("terminal accounting mismatch")
    return {
        "episodes": episodes,
        "agent_success_rate": success / (episodes * 5),
        "collision_rate": collision / (episodes * 5),
        "unresolved_rate": unresolved / (episodes * 5),
        "full_success_rate": sum(int(row[8]) for row in rows) / episodes,
        "timeout_episode_rate": sum(int(row[11]) for row in rows) / episodes,
        "mean_episode_steps": float(np.mean(rows[:, 3].astype(float))),
        "mean_interaction_share": float(np.mean(rows[:, 14].astype(float))),
        "mean_gate_switches": float(np.mean(rows[:, 15].astype(float))),
    }


def mcnemar(improved, degraded):
    count = improved + degraded
    if not count:
        return 1.0
    tail = sum(math.comb(count, k) for k in range(min(improved, degraded) + 1))
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
        "mcnemar_exact_p": mcnemar(improved, degraded),
    }


def main():
    with gzip.open(MANIFEST, "rt", encoding="utf-8") as handle:
        scenarios = json.load(handle)["scenarios"]
    expected = [str(item["scenario_id"]) for item in scenarios]
    strata = {str(item["scenario_id"]): item["view"]["g11_c_stratum"] for item in scenarios}
    runs = {}
    for repeat, seed in REPEATS.items():
        runs[repeat] = {}
        for policy in POLICIES:
            path = RESULT_DIR / f"g11_f_c_{policy}_r{repeat}_s{seed}.npy"
            rows = np.load(path, allow_pickle=True)
            if rows.shape != (50, 17) or [str(row[12]) for row in rows] != expected:
                raise ValueError(f"invalid or misordered result: {path}")
            runs[repeat][policy] = rows

    combined = {
        policy: np.concatenate([runs[repeat][policy] for repeat in REPEATS])
        for policy in POLICIES
    }
    summary = {
        "protocol": {
            "experiment_id": "G11-F-C-R2",
            "manifest": str(MANIFEST.relative_to(ROOT)),
            "episodes_per_run": 50,
            "repeats": REPEATS,
            "r2_actor": "capacity_wide_r2_s4_broad_n5_seed20260816_epoch_001",
            "sealed_test_read": False,
        },
        "per_repeat": {
            str(repeat): {policy: aggregate(runs[repeat][policy]) for policy in POLICIES}
            for repeat in REPEATS
        },
        "combined": {policy: aggregate(combined[policy]) for policy in POLICIES},
        "paired": {
            "a1_vs_5a": paired(combined["a1"], combined["5a"]),
            "b2_vs_a1": paired(combined["b2"], combined["a1"]),
            "a1_vs_r2": paired(combined["a1"], combined["r2"]),
            "r2_vs_5a": paired(combined["r2"], combined["5a"]),
        },
        "strata": {},
    }
    for name in sorted(set(strata.values())):
        mask = np.asarray([strata[str(row[12])] == name for row in combined["5a"]])
        summary["strata"][name] = {
            policy: aggregate(combined[policy][mask]) for policy in POLICIES
        }
    output = RUN_DIR / "local_data/pilot/summary_with_r2.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if ARCHIVE_LOG_DIR.exists():
        raise FileExistsError(f"archive already exists: {ARCHIVE_LOG_DIR}")
    ARCHIVE_LOG_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(ACTIVE_LOG_DIR), str(ARCHIVE_LOG_DIR))


if __name__ == "__main__":
    main()
