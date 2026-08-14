#!/usr/bin/env python3
import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np


def aggregate(rows):
    episodes = len(rows)
    agents = episodes * 5
    action_steps = sum(int(row[4]) for row in rows)
    interaction_steps = sum(float(row[14]) * int(row[4]) for row in rows)
    return {
        "episodes": episodes,
        "agent_success_rate": sum(int(row[6]) for row in rows) / agents,
        "collision_rate": sum(int(row[7]) for row in rows) / agents,
        "unresolved_rate": sum(int(row[10]) for row in rows) / agents,
        "full_success_rate": sum(int(row[8]) for row in rows) / episodes,
        "timeout_episode_rate": sum(int(row[11]) for row in rows) / episodes,
        "mean_episode_steps": float(np.mean([int(row[3]) for row in rows])),
        "mean_final_distance": float(np.mean([float(row[9]) for row in rows])),
        "interaction_action_share": interaction_steps / action_steps if action_steps else 0.0,
    }


def exact_mcnemar(improved, degraded):
    discordant = improved + degraded
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(improved, degraded) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def pairing(old_rows, new_rows):
    old_full = old_rows[:, 8].astype(int)
    new_full = new_rows[:, 8].astype(int)
    improved = int(np.sum(new_full > old_full))
    degraded = int(np.sum(new_full < old_full))
    return {
        "improved": improved,
        "degraded": degraded,
        "tied": len(old_rows) - improved - degraded,
        "mcnemar_exact_p": exact_mcnemar(improved, degraded),
    }


def load(path, expected_ids):
    rows = np.load(path, allow_pickle=True)
    if rows.shape != (len(expected_ids), 17):
        raise ValueError(f"invalid result shape for {path}: {rows.shape}")
    observed = [str(row[12]) for row in rows]
    if observed != expected_ids or len(set(observed)) != len(observed):
        raise ValueError(f"manifest order or uniqueness failed for {path}")
    if sum(int(row[6]) + int(row[7]) + int(row[10]) for row in rows) != len(rows) * 5:
        raise ValueError(f"terminal accounting failed for {path}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    args = parser.parse_args()

    with gzip.open(args.manifest, "rt", encoding="utf-8") as handle:
        scenarios = json.load(handle)["scenarios"]
    expected_ids = [str(item["scenario_id"]) for item in scenarios]
    topology = {
        str(item["scenario_id"]): str(item["view"]["capacity_topology"])
        for item in scenarios
    }

    by_seed = {}
    old_all, new_all = [], []
    for seed in args.seeds:
        old_rows = load(args.result_dir / f"avoidance_old_e16_s{seed}.npy", expected_ids)
        new_rows = load(args.result_dir / f"avoidance_candidate_e17_s{seed}.npy", expected_ids)
        old_all.append(old_rows)
        new_all.append(new_rows)
        by_seed[str(seed)] = {
            "epoch16": aggregate(old_rows),
            "epoch17": aggregate(new_rows),
            "paired": pairing(old_rows, new_rows),
        }

    old_all = np.concatenate(old_all)
    new_all = np.concatenate(new_all)
    pooled = {
        "epoch16": aggregate(old_all),
        "epoch17": aggregate(new_all),
        "paired": pairing(old_all, new_all),
    }
    by_topology = {}
    repeated_topology = np.asarray(
        [topology[item] for item in expected_ids] * len(args.seeds), dtype=object
    )
    for name in ("zero", "edge1", "multi"):
        mask = repeated_topology == name
        by_topology[name] = {
            "epoch16": aggregate(old_all[mask]),
            "epoch17": aggregate(new_all[mask]),
            "paired": pairing(old_all[mask], new_all[mask]),
        }

    old = pooled["epoch16"]
    new = pooled["epoch17"]
    topology_drops = {
        name: values["epoch17"]["full_success_rate"]
        - values["epoch16"]["full_success_rate"]
        for name, values in by_topology.items()
    }
    checks = {
        "full_success_higher": new["full_success_rate"] > old["full_success_rate"],
        "more_improved_than_degraded": pooled["paired"]["improved"]
        > pooled["paired"]["degraded"],
        "collision_not_higher": new["collision_rate"] <= old["collision_rate"],
        "timeout_delta_at_most_0_02": new["timeout_episode_rate"]
        - old["timeout_episode_rate"] <= 0.02,
        "mean_steps_ratio_at_most_1_10": new["mean_episode_steps"]
        <= old["mean_episode_steps"] * 1.10,
        "no_topology_full_success_drop_over_0_05": min(topology_drops.values()) >= -0.05,
    }
    summary = {
        "protocol": {
            "manifest": str(args.manifest),
            "episodes_per_policy_seed": len(expected_ids),
            "seeds": args.seeds,
            "oracle_distance_m": 2.0,
        },
        "audit": {"manifest_order": "passed", "duplicate_ids": 0},
        "by_seed": by_seed,
        "pooled": pooled,
        "by_topology": by_topology,
        "topology_full_success_delta": topology_drops,
        "admission_checks": checks,
        "admitted": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
