#!/usr/bin/env python3
import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np


COLUMNS = {
    "episode": 0,
    "total_env_steps": 1,
    "total_agent_samples": 2,
    "episode_env_steps": 3,
    "episode_agent_samples": 4,
    "mean_reward": 5,
    "success": 6,
    "collision": 7,
    "full_success": 8,
    "mean_final_distance": 9,
    "unresolved": 10,
    "timeout": 11,
    "scenario_id": 12,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two same-scenario Actor validation result arrays."
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_stats(path):
    rows = np.load(path, allow_pickle=True)
    if rows.ndim != 2 or rows.shape[1] != len(COLUMNS):
        raise ValueError(
            f"Unexpected stats shape for {path}: {rows.shape}; expected (N, 13)"
        )
    scenario_ids = [str(row[COLUMNS["scenario_id"]]) for row in rows]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError(f"Duplicate scenario IDs in {path}")
    return rows


def exact_mcnemar_p(baseline_only, candidate_only):
    discordant = baseline_only + candidate_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k) for k in range(min(baseline_only, candidate_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def aggregate(rows):
    episodes = len(rows)
    agents = episodes * 5
    timeout_mask = rows[:, COLUMNS["timeout"]].astype(int) == 1
    completed_steps = rows[~timeout_mask, COLUMNS["episode_env_steps"]].astype(float)
    return {
        "episodes": episodes,
        "agent_success": float(rows[:, COLUMNS["success"]].astype(float).sum() / agents),
        "collision": float(rows[:, COLUMNS["collision"]].astype(float).sum() / agents),
        "unresolved": float(rows[:, COLUMNS["unresolved"]].astype(float).sum() / agents),
        "full_success_count": int(rows[:, COLUMNS["full_success"]].astype(int).sum()),
        "full_success": float(
            rows[:, COLUMNS["full_success"]].astype(float).mean()
        ),
        "timeout_count": int(timeout_mask.sum()),
        "timeout": float(timeout_mask.mean()),
        "mean_env_steps": float(
            rows[:, COLUMNS["episode_env_steps"]].astype(float).mean()
        ),
        "mean_env_steps_non_timeout": (
            float(completed_steps.mean()) if len(completed_steps) else None
        ),
    }


def load_conflict_edges(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    result = {}
    for scenario in payload["scenarios"]:
        scenario_id = str(scenario["scenario_id"])
        result[scenario_id] = len(scenario.get("metrics", {}).get("conflict_edges", []))
    return result


def edge_band(edge_count):
    return str(edge_count) if edge_count < 3 else "3+"


def main():
    args = parse_args()
    baseline = load_stats(args.baseline)
    candidate = load_stats(args.candidate)

    baseline_ids = [str(row[COLUMNS["scenario_id"]]) for row in baseline]
    candidate_ids = [str(row[COLUMNS["scenario_id"]]) for row in candidate]
    if baseline_ids != candidate_ids:
        raise ValueError("Baseline and candidate scenario IDs/order do not match")

    baseline_full = baseline[:, COLUMNS["full_success"]].astype(bool)
    candidate_full = candidate[:, COLUMNS["full_success"]].astype(bool)
    baseline_timeout = baseline[:, COLUMNS["timeout"]].astype(bool)
    candidate_timeout = candidate[:, COLUMNS["timeout"]].astype(bool)

    both_full = int(np.sum(baseline_full & candidate_full))
    baseline_only = int(np.sum(baseline_full & ~candidate_full))
    candidate_only = int(np.sum(~baseline_full & candidate_full))
    neither_full = int(np.sum(~baseline_full & ~candidate_full))

    baseline_success = baseline[:, COLUMNS["success"]].astype(int)
    candidate_success = candidate[:, COLUMNS["success"]].astype(int)
    success_delta = candidate_success - baseline_success

    baseline_aggregate = aggregate(baseline)
    candidate_aggregate = aggregate(candidate)
    result = {
        "result_version": 1,
        "status": "complete",
        "protocol": {
            "episodes": len(baseline),
            "baseline": args.baseline_label,
            "candidate": args.candidate_label,
            "paired_scenario_order": True,
            "baseline_stats": str(args.baseline),
            "candidate_stats": str(args.candidate),
            "manifest": str(args.manifest) if args.manifest else None,
        },
        args.baseline_label: baseline_aggregate,
        args.candidate_label: candidate_aggregate,
        "delta_candidate_minus_baseline": {
            key: candidate_aggregate[key] - baseline_aggregate[key]
            for key in (
                "agent_success",
                "collision",
                "unresolved",
                "full_success",
                "timeout",
                "mean_env_steps",
                "mean_env_steps_non_timeout",
            )
        },
        "paired_full_success": {
            "both": both_full,
            "baseline_only": baseline_only,
            "candidate_only": candidate_only,
            "neither": neither_full,
            "oracle_union_count": both_full + baseline_only + candidate_only,
            "oracle_union": float(
                (both_full + baseline_only + candidate_only) / len(baseline)
            ),
            "mcnemar_exact_p": exact_mcnemar_p(baseline_only, candidate_only),
        },
        "paired_agent_success_count": {
            "candidate_better_cases": int(np.sum(success_delta > 0)),
            "equal_cases": int(np.sum(success_delta == 0)),
            "candidate_worse_cases": int(np.sum(success_delta < 0)),
            "net_additional_agent_successes": int(success_delta.sum()),
        },
        "paired_timeout": {
            "both": int(np.sum(baseline_timeout & candidate_timeout)),
            "baseline_only": int(np.sum(baseline_timeout & ~candidate_timeout)),
            "candidate_only": int(np.sum(~baseline_timeout & candidate_timeout)),
            "neither": int(np.sum(~baseline_timeout & ~candidate_timeout)),
            "candidate_timeout_baseline_full": int(
                np.sum(candidate_timeout & baseline_full)
            ),
        },
    }

    if args.manifest:
        conflicts = load_conflict_edges(args.manifest)
        missing = [scenario_id for scenario_id in baseline_ids if scenario_id not in conflicts]
        if missing:
            raise ValueError(f"Manifest is missing {len(missing)} evaluated scenario IDs")
        stratified = {}
        bands = np.array([edge_band(conflicts[item]) for item in baseline_ids])
        for band in ("0", "1", "2", "3+"):
            mask = bands == band
            if not np.any(mask):
                continue
            stratified[band] = {
                "scenarios": int(mask.sum()),
                args.baseline_label: aggregate(baseline[mask]),
                args.candidate_label: aggregate(candidate[mask]),
            }
        result["by_conflict_edges"] = stratified

    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
