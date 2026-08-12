#!/usr/bin/env python3
import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np


def load_run(path, expected_ids):
    rows = np.load(path, allow_pickle=True)
    if rows.shape != (len(expected_ids), 17):
        raise ValueError(f"invalid result shape for {path}: {rows.shape}")
    observed = [str(row[12]) for row in rows]
    if observed != expected_ids:
        raise ValueError(f"manifest order mismatch for {path}")
    if len(set(observed)) != len(observed):
        raise ValueError(f"duplicate scenario IDs for {path}")
    return rows


def aggregate(rows):
    episodes = len(rows)
    if episodes == 0:
        return {
            "episodes": 0,
            "agent_success_rate": None,
            "collision_rate": None,
            "unresolved_rate": None,
            "full_success_rate": None,
            "timeout_episode_rate": None,
            "mean_episode_steps": None,
            "mean_final_distance": None,
            "interaction_action_share": None,
            "interaction_episode_rate": None,
        }
    successes = sum(int(row[6]) for row in rows)
    collisions = sum(int(row[7]) for row in rows)
    unresolved = sum(int(row[10]) for row in rows)
    if successes + collisions + unresolved != episodes * 5:
        raise ValueError("terminal outcome accounting mismatch")
    action_steps = sum(int(row[4]) for row in rows)
    interaction_steps = sum(float(row[14]) * int(row[4]) for row in rows)
    return {
        "episodes": episodes,
        "agent_success_rate": successes / (episodes * 5),
        "collision_rate": collisions / (episodes * 5),
        "unresolved_rate": unresolved / (episodes * 5),
        "full_success_rate": sum(int(row[8]) for row in rows) / episodes,
        "timeout_episode_rate": sum(int(row[11]) for row in rows) / episodes,
        "mean_episode_steps": float(np.mean([int(row[3]) for row in rows])),
        "mean_final_distance": float(np.mean([float(row[9]) for row in rows])),
        "interaction_action_share": interaction_steps / action_steps if action_steps else 0.0,
        "interaction_episode_rate": sum(float(row[14]) > 0.0 for row in rows) / episodes,
    }


def mcnemar_exact(improved, degraded):
    discordant = improved + degraded
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(improved, degraded) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def paired(candidate, baseline):
    improved = sum(int(a[8]) > int(b[8]) for a, b in zip(candidate, baseline))
    degraded = sum(int(a[8]) < int(b[8]) for a, b in zip(candidate, baseline))
    return {
        "full_success_improved": improved,
        "full_success_degraded": degraded,
        "full_success_tied": len(candidate) - improved - degraded,
        "mcnemar_exact_p": mcnemar_exact(improved, degraded),
    }


def subset(rows, expected_ids, selected_ids):
    by_id = {scenario_id: row for scenario_id, row in zip(expected_ids, rows)}
    return np.asarray([by_id[item] for item in expected_ids if item in selected_ids], dtype=object)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--e2", required=True, type=Path)
    parser.add_argument("--old-recovery", required=True, type=Path)
    parser.add_argument("--new-recovery", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()

    with gzip.open(args.manifest, "rt", encoding="utf-8") as handle:
        scenarios = json.load(handle)["scenarios"]
    expected_ids = [str(item["scenario_id"]) for item in scenarios]
    metadata = {
        str(item["scenario_id"]): {
            "pool": str(item["view"].get("capacity_pool", item["view"].get("ie2_pool", "unknown"))),
            "topology": str(item["view"].get("capacity_topology", item["view"].get("ie2_topology", "unknown"))),
        }
        for item in scenarios
    }
    runs = {
        "e2_matched": load_run(args.e2, expected_ids),
        "e2_old_epoch16_recovery": load_run(args.old_recovery, expected_ids),
    }
    if args.new_recovery:
        runs["e2_ie2_recovery"] = load_run(args.new_recovery, expected_ids)

    groups = {
        "overall": {"all": set(expected_ids)},
        "by_pool": {
            name: {key for key, value in metadata.items() if value["pool"] == name}
            for name in ("standard", "dense")
        },
        "by_topology": {
            name: {key for key, value in metadata.items() if value["topology"] == name}
            for name in ("zero", "edge1", "edge2", "edge3plus", "multi")
        },
    }
    metrics = {
        dimension: {
            name: {
                policy: aggregate(subset(rows, expected_ids, ids))
                for policy, rows in runs.items()
            }
            for name, ids in dimension_groups.items()
        }
        for dimension, dimension_groups in groups.items()
    }
    pairings = {
        f"{policy}_vs_e2_matched": {
            dimension: {
                name: paired(
                    subset(rows, expected_ids, ids),
                    subset(runs["e2_matched"], expected_ids, ids),
                )
                for name, ids in dimension_groups.items()
            }
            for dimension, dimension_groups in groups.items()
        }
        for policy, rows in runs.items()
        if policy != "e2_matched"
    }
    payload = {
        "protocol": {
            "experiment": "e2-ie2-matched-overnight-pipeline",
            "manifest": str(args.manifest),
            "seed": args.seed,
            "policies": list(runs),
        },
        "audit": {"manifest_order": "passed", "duplicate_ids": 0, "missing_ids": 0},
        "metrics": metrics,
        "paired": pairings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
