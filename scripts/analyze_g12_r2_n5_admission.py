#!/usr/bin/env python3
import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np


def aggregate(rows):
    episodes = len(rows)
    if episodes == 0:
        raise ValueError("cannot aggregate an empty subset")
    successes = sum(int(row[6]) for row in rows)
    collisions = sum(int(row[7]) for row in rows)
    unresolved = sum(int(row[10]) for row in rows)
    if successes + collisions + unresolved != episodes * 5:
        raise ValueError("terminal outcome accounting mismatch")
    return {
        "episodes": episodes,
        "agent_success_rate": successes / (episodes * 5),
        "collision_rate": collisions / (episodes * 5),
        "unresolved_rate": unresolved / (episodes * 5),
        "full_success_rate": sum(int(row[8]) for row in rows) / episodes,
        "timeout_episode_rate": sum(int(row[11]) for row in rows) / episodes,
        "mean_episode_steps": float(np.mean([int(row[3]) for row in rows])),
        "mean_final_distance": float(np.mean([float(row[9]) for row in rows])),
    }


def mcnemar_exact(improved, degraded):
    discordant = improved + degraded
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(improved, degraded) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def paired(candidate, baseline):
    candidate_by_id = {str(row[12]): row for row in candidate}
    baseline_by_id = {str(row[12]): row for row in baseline}
    if set(candidate_by_id) != set(baseline_by_id):
        raise ValueError("paired runs contain different scenario IDs")
    improved = sum(
        int(candidate_by_id[key][8]) > int(baseline_by_id[key][8])
        for key in candidate_by_id
    )
    degraded = sum(
        int(candidate_by_id[key][8]) < int(baseline_by_id[key][8])
        for key in candidate_by_id
    )
    return {
        "full_success_improved": improved,
        "full_success_degraded": degraded,
        "full_success_tied": len(candidate_by_id) - improved - degraded,
        "mcnemar_exact_p": mcnemar_exact(improved, degraded),
    }


def subset(rows, ids):
    return np.asarray([row for row in rows if str(row[12]) in ids], dtype=object)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--candidate-policy", default="r2")
    args = parser.parse_args()
    policies = ("5a", args.candidate_policy)

    with gzip.open(args.manifest, "rt", encoding="utf-8") as handle:
        scenarios = json.load(handle)["scenarios"]
    expected_ids = [str(item["scenario_id"]) for item in scenarios]
    metadata = {
        str(item["scenario_id"]): {
            "pool": str(item["view"]["capacity_pool"]),
            "topology": str(item["view"]["capacity_topology"]),
            "stratum": str(item["view"]["g12_stratum"]),
        }
        for item in scenarios
    }

    runs = {}
    for policy in policies:
        path = args.results_dir / f"g12_r2_n5_admission_{policy}_s{args.seed}.npy"
        rows = np.load(path, allow_pickle=True)
        if rows.shape != (len(expected_ids), 17):
            raise ValueError(f"invalid result shape for {policy}: {rows.shape}")
        observed = [str(row[12]) for row in rows]
        if observed != expected_ids:
            raise ValueError(f"manifest order mismatch for {policy}")
        if len(set(observed)) != len(observed):
            raise ValueError(f"duplicate scenario IDs for {policy}")
        runs[policy] = rows

    dimensions = {
        "overall": {"all": set(expected_ids)},
        "by_pool": {
            name: {key for key, value in metadata.items() if value["pool"] == name}
            for name in ("standard", "dense")
        },
        "by_topology": {
            name: {
                key for key, value in metadata.items() if value["topology"] == name
            }
            for name in ("zero", "edge1", "multi")
        },
        "by_stratum": {
            name: {key for key, value in metadata.items() if value["stratum"] == name}
            for name in sorted({value["stratum"] for value in metadata.values()})
        },
    }
    metrics = {
        dimension: {
            name: {
                policy: aggregate(subset(rows, ids))
                for policy, rows in runs.items()
            }
            for name, ids in groups.items()
        }
        for dimension, groups in dimensions.items()
    }
    pairing = {
        dimension: {
            name: paired(
                subset(runs[args.candidate_policy], ids), subset(runs["5a"], ids)
            )
            for name, ids in groups.items()
        }
        for dimension, groups in dimensions.items()
    }

    overall = metrics["overall"]["all"]
    zero = metrics["by_topology"]["zero"]
    pools = metrics["by_pool"]
    checks = {
        "zero_full_success_drop_at_most_0.03": (
            zero[args.candidate_policy]["full_success_rate"]
            >= zero["5a"]["full_success_rate"] - 0.03
        ),
        "overall_agent_success_drop_at_most_0.02": (
            overall[args.candidate_policy]["agent_success_rate"]
            >= overall["5a"]["agent_success_rate"] - 0.02
        ),
        "overall_timeout_increase_at_most_0.02": (
            overall[args.candidate_policy]["timeout_episode_rate"]
            <= overall["5a"]["timeout_episode_rate"] + 0.02
        ),
        "standard_full_success_drop_at_most_0.05": (
            pools["standard"][args.candidate_policy]["full_success_rate"]
            >= pools["standard"]["5a"]["full_success_rate"] - 0.05
        ),
        "dense_full_success_drop_at_most_0.05": (
            pools["dense"][args.candidate_policy]["full_success_rate"]
            >= pools["dense"]["5a"]["full_success_rate"] - 0.05
        ),
    }
    summary = {
        "protocol": {
            "experiment": "G12-R2-N5-admission",
            "manifest": str(args.manifest),
            "episodes_per_policy": len(expected_ids),
            "seed": args.seed,
            "policies": list(policies),
        },
        "audit": {
            "manifest_order": "passed",
            "duplicate_ids": 0,
            "missing_ids": 0,
        },
        "metrics": metrics,
        "paired": pairing,
        "admission_checks": checks,
        "admission_passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
