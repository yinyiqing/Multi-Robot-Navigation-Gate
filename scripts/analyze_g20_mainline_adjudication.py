#!/usr/bin/env python3
import gzip
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/03_保留专门化/02_论文主线"
RUN_DIR = BASE / "20_夜间最终统一评测/local_data"
MANIFEST = BASE / "datasets/fixed_v1/views/g12_full_scene_selection_v1/validation.json.gz"
POLICIES = ("n5", "r2", "f_a1", "n5_recovery")
SEEDS = (20260830, 20260831)


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
        "full_success_rate": sum(int(row[8]) for row in rows) / episodes,
        "collision_rate": collision / (episodes * 5),
        "unresolved_rate": unresolved / (episodes * 5),
        "timeout_episode_rate": sum(int(row[11]) for row in rows) / episodes,
        "mean_episode_steps": float(np.mean(rows[:, 3].astype(float))),
        "mean_interaction_share": float(np.mean(rows[:, 14].astype(float))),
        "mean_gate_switches": float(np.mean(rows[:, 15].astype(float))),
    }


def mcnemar(improved, degraded):
    discordant = improved + degraded
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(improved, degraded) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def paired(candidate, baseline):
    candidate_full = candidate[:, 8].astype(int)
    baseline_full = baseline[:, 8].astype(int)
    improved = int((candidate_full > baseline_full).sum())
    degraded = int((candidate_full < baseline_full).sum())
    return {
        "improved": improved,
        "degraded": degraded,
        "tied": len(candidate_full) - improved - degraded,
        "mcnemar_exact_p": mcnemar(improved, degraded),
    }


def topology(stratum):
    if stratum.endswith("-zero"):
        return "zero"
    if stratum.endswith("-edge1"):
        return "edge-1"
    if stratum.endswith("-multi"):
        return "multi-edge"
    raise ValueError(f"unknown G12 stratum: {stratum}")


def main():
    with gzip.open(MANIFEST, "rt", encoding="utf-8") as handle:
        scenarios = json.load(handle)["scenarios"]
    ids = [str(item["scenario_id"]) for item in scenarios]
    layers = {
        str(item["scenario_id"]): topology(str(item["view"]["g12_stratum"]))
        for item in scenarios
    }

    runs = {}
    for seed in SEEDS:
        runs[seed] = {}
        for policy in POLICIES:
            path = RUN_DIR / "results" / f"g20_{policy}_s{seed}.npy"
            rows = np.load(path, allow_pickle=True)
            observed_ids = [str(row[12]) for row in rows]
            if rows.shape != (120, 17) or observed_ids != ids:
                raise ValueError(f"invalid or misordered result: {path}")
            aggregate(rows)
            runs[seed][policy] = rows

    combined = {
        policy: np.concatenate([runs[seed][policy] for seed in SEEDS])
        for policy in POLICIES
    }
    summary = {
        "protocol": {
            "experiment_id": "G20-mainline-adjudication",
            "manifest_sha256": "52435d6c5bdf9914e7212dd29cb4bfec074257f72d85f0d71741deee7c63b635",
            "unique_scenarios": 120,
            "topology_counts_per_seed": {"zero": 40, "edge-1": 40, "multi-edge": 40},
            "seeds": SEEDS,
            "sealed_test_read": False,
        },
        "per_seed": {
            str(seed): {policy: aggregate(runs[seed][policy]) for policy in POLICIES}
            for seed in SEEDS
        },
        "combined": {policy: aggregate(combined[policy]) for policy in POLICIES},
        "paired_vs_n5": {
            policy: paired(combined[policy], combined["n5"])
            for policy in POLICIES
            if policy != "n5"
        },
        "topology": {},
    }
    for layer in ("zero", "edge-1", "multi-edge"):
        mask = np.asarray([layers[str(row[12])] == layer for row in combined["n5"]])
        summary["topology"][layer] = {
            "metrics": {policy: aggregate(combined[policy][mask]) for policy in POLICIES},
            "paired_vs_n5": {
                policy: paired(combined[policy][mask], combined["n5"][mask])
                for policy in POLICIES
                if policy != "n5"
            },
        }

    output = RUN_DIR / "summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
