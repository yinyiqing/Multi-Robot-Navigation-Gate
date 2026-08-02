#!/usr/bin/env python3
"""Analyze the frozen exact-edge-2 zero-shot confirmation."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


COLUMNS = {
    "episode": 0,
    "episode_env_steps": 3,
    "success": 6,
    "collision": 7,
    "full_success": 8,
    "unresolved": 10,
    "timeout": 11,
    "scenario_id": 12,
    "dense_action_share": 14,
    "gate_switches": 15,
    "gate_mean_probability": 16,
}
TOPOLOGY_KEYS = ("max_conflict_degree", "simultaneous_conflict_count")
BASELINE_RUN = "edge2_5a_n200_s20260803_20260802_040932"
GATE_RUN = "edge2_learned_gate_n200_s20260803_20260802_123750"


def parse_args():
    route_dir = Path(__file__).resolve().parent
    main_dir = route_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=route_dir / "local_data/5a/results" / f"{BASELINE_RUN}.npy",
    )
    parser.add_argument(
        "--gate",
        type=Path,
        default=route_dir
        / "local_data/learned-gate/results"
        / f"{GATE_RUN}.npy",
    )
    parser.add_argument(
        "--oracle",
        type=Path,
        default=main_dir
        / "results/05_当前冻结方案/D4_dense_validation_actor_comparison_s20260728"
        / "5a_plus_epoch16_oracle_evaluation.npy",
    )
    parser.add_argument(
        "--historical-baseline",
        type=Path,
        default=main_dir
        / "results/05_当前冻结方案/D4_dense_validation_actor_comparison_s20260728"
        / "5a_evaluation.npy",
    )
    parser.add_argument("--manifest", type=Path, default=route_dir / "validation.json")
    parser.add_argument("--output", type=Path, default=route_dir / "summary.json")
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260802)
    return parser.parse_args()


def load_rows(path, minimum_columns):
    rows = np.load(path, allow_pickle=True)
    if rows.ndim != 2 or rows.shape[1] < minimum_columns:
        raise ValueError(
            f"Unexpected stats shape for {path}: {rows.shape}; "
            f"expected at least {minimum_columns} columns"
        )
    ids = [str(row[COLUMNS["scenario_id"]]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate scenario IDs in {path}")
    return rows


def select_by_id(rows, scenario_ids, label):
    row_by_id = {str(row[COLUMNS["scenario_id"]]): row for row in rows}
    missing = [scenario_id for scenario_id in scenario_ids if scenario_id not in row_by_id]
    if missing:
        raise ValueError(f"{label} is missing {len(missing)} scenario IDs")
    return np.asarray([row_by_id[scenario_id] for scenario_id in scenario_ids], dtype=object)


def aggregate(rows, include_gate=False):
    episodes = len(rows)
    result = {
        "episodes": episodes,
        "agent_success_count": int(rows[:, COLUMNS["success"]].astype(int).sum()),
        "agent_success": float(
            rows[:, COLUMNS["success"]].astype(float).sum() / (episodes * 5)
        ),
        "collision_count": int(rows[:, COLUMNS["collision"]].astype(int).sum()),
        "collision": float(
            rows[:, COLUMNS["collision"]].astype(float).sum() / (episodes * 5)
        ),
        "unresolved_count": int(rows[:, COLUMNS["unresolved"]].astype(int).sum()),
        "unresolved": float(
            rows[:, COLUMNS["unresolved"]].astype(float).sum() / (episodes * 5)
        ),
        "full_success_count": int(
            rows[:, COLUMNS["full_success"]].astype(int).sum()
        ),
        "full_success": float(rows[:, COLUMNS["full_success"]].astype(float).mean()),
        "timeout_count": int(rows[:, COLUMNS["timeout"]].astype(int).sum()),
        "timeout": float(rows[:, COLUMNS["timeout"]].astype(float).mean()),
        "mean_env_steps": float(
            rows[:, COLUMNS["episode_env_steps"]].astype(float).mean()
        ),
    }
    if include_gate:
        result.update(
            {
                "mean_dense_action_share": float(
                    rows[:, COLUMNS["dense_action_share"]].astype(float).mean()
                ),
                "mean_gate_switches": float(
                    rows[:, COLUMNS["gate_switches"]].astype(float).mean()
                ),
                "mean_gate_probability": float(
                    rows[:, COLUMNS["gate_mean_probability"]].astype(float).mean()
                ),
            }
        )
    return result


def exact_mcnemar_p(baseline_only, gate_only):
    discordant = baseline_only + gate_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k)
        for k in range(min(baseline_only, gate_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def paired_comparison(baseline, gate):
    baseline_full = baseline[:, COLUMNS["full_success"]].astype(bool)
    gate_full = gate[:, COLUMNS["full_success"]].astype(bool)
    baseline_only = int(np.sum(baseline_full & ~gate_full))
    gate_only = int(np.sum(~baseline_full & gate_full))
    return {
        "both_success": int(np.sum(baseline_full & gate_full)),
        "gate_only_success": gate_only,
        "baseline_only_success": baseline_only,
        "both_failure": int(np.sum(~baseline_full & ~gate_full)),
        "mcnemar_exact_p": exact_mcnemar_p(baseline_only, gate_only),
    }


def paired_bootstrap(baseline, gate, resamples, seed):
    delta = (
        gate[:, COLUMNS["full_success"]].astype(float)
        - baseline[:, COLUMNS["full_success"]].astype(float)
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(resamples, len(delta)))
    samples = delta[indices].mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return {
        "metric": "full_success_delta",
        "resamples": resamples,
        "seed": seed,
        "mean": float(delta.mean()),
        "ci95": [float(low), float(high)],
    }


def main():
    args = parse_args()
    baseline = load_rows(args.baseline, 17)
    gate = load_rows(args.gate, 17)
    if len(baseline) != 200 or len(gate) != 200:
        raise ValueError("Confirmation requires exactly 200 rows per deployed method")

    with args.manifest.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    scenarios = manifest["scenarios"]
    scenario_ids = [str(case["scenario_id"]) for case in scenarios]
    baseline_ids = [str(row[COLUMNS["scenario_id"]]) for row in baseline]
    gate_ids = [str(row[COLUMNS["scenario_id"]]) for row in gate]
    if baseline_ids != scenario_ids or gate_ids != scenario_ids:
        raise ValueError("Deployed result IDs/order do not match the frozen manifest")

    oracle = select_by_id(load_rows(args.oracle, 13), scenario_ids, "oracle")
    historical_baseline = select_by_id(
        load_rows(args.historical_baseline, 13), scenario_ids, "historical baseline"
    )
    topology = {
        str(case["scenario_id"]): case.get("metrics", {}) for case in scenarios
    }

    baseline_summary = aggregate(baseline)
    gate_summary = aggregate(gate, include_gate=True)
    oracle_summary = aggregate(oracle)
    bootstrap = paired_bootstrap(
        baseline, gate, args.bootstrap_resamples, args.bootstrap_seed
    )
    paired = paired_comparison(baseline, gate)
    full_success_delta = gate_summary["full_success"] - baseline_summary["full_success"]
    oracle_gain = oracle_summary["full_success"] - baseline_summary["full_success"]
    oracle_gain_recovery = full_success_delta / oracle_gain if oracle_gain > 0 else None

    strata = {}
    for key in TOPOLOGY_KEYS:
        values = np.asarray([int(topology[scenario_id].get(key, 0)) for scenario_id in scenario_ids])
        strata[key] = {}
        for value in sorted(set(values.tolist())):
            mask = values == value
            strata[key][str(value)] = {
                "scenarios": int(mask.sum()),
                "baseline": aggregate(baseline[mask]),
                "learned_gate": aggregate(gate[mask], include_gate=True),
                "oracle": aggregate(oracle[mask]),
            }

    criteria = {
        "full_success_delta_at_least_0_08": full_success_delta >= 0.08 - 1e-12,
        "bootstrap_ci_lower_above_zero": bootstrap["ci95"][0] > 0.0,
        "mcnemar_p_below_0_05": paired["mcnemar_exact_p"] < 0.05,
        "oracle_gain_recovery_at_least_0_60": (
            oracle_gain_recovery is not None and oracle_gain_recovery >= 0.60
        ),
        "agent_success_not_lower": (
            gate_summary["agent_success"] >= baseline_summary["agent_success"]
        ),
        "timeout_increase_at_most_0_01": (
            gate_summary["timeout"] - baseline_summary["timeout"] <= 0.01
        ),
    }
    result = {
        "result_version": 1,
        "status": "passed" if all(criteria.values()) else "failed",
        "integrity": {
            "expected_episodes": 200,
            "unique_scenario_ids": len(set(scenario_ids)),
            "manifest_order_matches": True,
        },
        "overall": {
            "baseline": baseline_summary,
            "learned_gate": gate_summary,
            "historical_baseline_same_ids": aggregate(historical_baseline),
            "historical_oracle_same_ids": oracle_summary,
        },
        "paired_full_success": paired,
        "paired_bootstrap": bootstrap,
        "full_success_delta": full_success_delta,
        "oracle_gain": oracle_gain,
        "oracle_gain_recovery": oracle_gain_recovery,
        "criteria": criteria,
        "strata": strata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
