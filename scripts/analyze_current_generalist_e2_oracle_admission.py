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
    # The current 17-column test schema stores per-episode interaction share in
    # column 14 and active action count in column 4. Columns 13 and 15 are the
    # rule-enabled flag and Gate switch count, respectively.
    action_steps = sum(int(row[4]) for row in rows)
    dense_action_steps = sum(float(row[14]) * int(row[4]) for row in rows)
    return {
        "episodes": episodes,
        "agent_success_rate": successes / (episodes * 5),
        "collision_rate": collisions / (episodes * 5),
        "unresolved_rate": unresolved / (episodes * 5),
        "full_success_rate": sum(int(row[8]) for row in rows) / episodes,
        "timeout_episode_rate": sum(int(row[11]) for row in rows) / episodes,
        "mean_episode_steps": float(np.mean([int(row[3]) for row in rows])),
        "mean_final_distance": float(np.mean([float(row[9]) for row in rows])),
        "interaction_action_share": (
            dense_action_steps / action_steps if action_steps else 0.0
        ),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--five-a-result", required=True, type=Path)
    parser.add_argument("--r2-result", required=True, type=Path)
    parser.add_argument("--n5-result", required=True, type=Path)
    parser.add_argument("--e2-result", required=True, type=Path)
    parser.add_argument("--e2-oracle-result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--candidate-key",
        default="e2_oracle_epoch16",
        help="Metric key for the candidate dual-actor run.",
    )
    parser.add_argument(
        "--experiment-name",
        default="current-generalist-E2-oracle-epoch16-admission",
    )
    parser.add_argument(
        "--candidate-description",
        default="interaction_oracle distance <= 2.0 m, standard=E2, interaction=epoch16",
    )
    args = parser.parse_args()
    candidate_key = str(args.candidate_key)

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

    runs = {
        "5a": load_run(args.five_a_result, expected_ids),
        "r2_10k": load_run(args.r2_result, expected_ids),
        "n5_20k": load_run(args.n5_result, expected_ids),
        "e2": load_run(args.e2_result, expected_ids),
        candidate_key: load_run(args.e2_oracle_result, expected_ids),
    }
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
            name: {policy: aggregate(subset(rows, ids)) for policy, rows in runs.items()}
            for name, ids in groups.items()
        }
        for dimension, groups in dimensions.items()
    }
    pairing = {
        f"{candidate_key}_vs_{baseline}": {
            dimension: {
                name: paired(
                    subset(runs[candidate_key], ids),
                    subset(runs[baseline], ids),
                )
                for name, ids in groups.items()
            }
            for dimension, groups in dimensions.items()
        }
        for baseline in ("5a", "r2_10k", "n5_20k", "e2")
    }
    overall = metrics["overall"]["all"]
    summary = {
        "protocol": {
            "experiment": str(args.experiment_name),
            "manifest": str(args.manifest),
            "episodes_per_policy": len(expected_ids),
            "seed": args.seed,
            "policies": list(runs),
            "oracle": str(args.candidate_description),
        },
        "audit": {
            "manifest_order": "passed",
            "duplicate_ids": 0,
            "missing_ids": 0,
        },
        "metrics": metrics,
        "paired": pairing,
        f"{candidate_key}_vs_e2_delta": {
            key: (
                overall[candidate_key][key] - overall["e2"][key]
                if isinstance(overall[candidate_key][key], (int, float))
                else None
            )
            for key in overall[candidate_key]
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
