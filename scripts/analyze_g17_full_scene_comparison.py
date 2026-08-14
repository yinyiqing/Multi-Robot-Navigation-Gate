#!/usr/bin/env python3
import gzip
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/03_保留专门化/02_论文主线"
RUN_DIR = BASE / "17_完整场景统一对比/local_data"
MANIFEST = BASE / "datasets/fixed_v1/views/g12_full_scene_selection_v1/validation.json.gz"
POLICIES = ("5a", "r2bbest", "a1")
SEEDS = (20260824, 20260825)


def aggregate(rows):
    episodes = len(rows)
    success = sum(int(x[6]) for x in rows)
    collision = sum(int(x[7]) for x in rows)
    unresolved = sum(int(x[10]) for x in rows)
    if success + collision + unresolved != episodes * 5:
        raise ValueError("terminal accounting mismatch")
    return {
        "episodes": episodes,
        "agent_success_rate": success / (episodes * 5),
        "full_success_rate": sum(int(x[8]) for x in rows) / episodes,
        "collision_rate": collision / (episodes * 5),
        "unresolved_rate": unresolved / (episodes * 5),
        "timeout_episode_rate": sum(int(x[11]) for x in rows) / episodes,
        "mean_episode_steps": float(np.mean(rows[:, 3].astype(float))),
        "mean_interaction_share": float(np.mean(rows[:, 14].astype(float))),
        "mean_gate_switches": float(np.mean(rows[:, 15].astype(float))),
    }


def mcnemar(improved, degraded):
    n = improved + degraded
    if not n:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(improved, degraded) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def paired(candidate, baseline):
    c, b = candidate[:, 8].astype(int), baseline[:, 8].astype(int)
    improved, degraded = int((c > b).sum()), int((c < b).sum())
    return {
        "improved": improved,
        "degraded": degraded,
        "tied": len(c) - improved - degraded,
        "mcnemar_exact_p": mcnemar(improved, degraded),
    }


def main():
    with gzip.open(MANIFEST, "rt", encoding="utf-8") as handle:
        scenarios = json.load(handle)["scenarios"]
    ids = [str(x["scenario_id"]) for x in scenarios]
    strata = {str(x["scenario_id"]): str(x["view"]["g12_stratum"]) for x in scenarios}
    runs = {}
    for seed in SEEDS:
        runs[seed] = {}
        for policy in POLICIES:
            path = RUN_DIR / "results" / f"g17_{policy}_s{seed}.npy"
            rows = np.load(path, allow_pickle=True)
            if rows.shape != (120, 17) or [str(x[12]) for x in rows] != ids:
                raise ValueError(f"invalid result: {path}")
            runs[seed][policy] = rows
    combined = {
        p: np.concatenate([runs[s][p] for s in SEEDS]) for p in POLICIES
    }
    summary = {
        "protocol": {
            "experiment_id": "G17-full-scene-comparison",
            "manifest_sha256": "52435d6c5bdf9914e7212dd29cb4bfec074257f72d85f0d71741deee7c63b635",
            "unique_scenarios": 120,
            "seeds": SEEDS,
            "sealed_test_read": False,
        },
        "per_seed": {
            str(s): {p: aggregate(runs[s][p]) for p in POLICIES} for s in SEEDS
        },
        "combined": {p: aggregate(combined[p]) for p in POLICIES},
        "paired": {
            "r2bbest_vs_5a": paired(combined["r2bbest"], combined["5a"]),
            "a1_vs_r2bbest": paired(combined["a1"], combined["r2bbest"]),
            "a1_vs_5a": paired(combined["a1"], combined["5a"]),
        },
        "strata": {},
    }
    for name in sorted(set(strata.values())):
        mask = np.asarray([strata[str(x[12])] == name for x in combined["5a"]])
        summary["strata"][name] = {p: aggregate(combined[p][mask]) for p in POLICIES}
    output = RUN_DIR / "summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
