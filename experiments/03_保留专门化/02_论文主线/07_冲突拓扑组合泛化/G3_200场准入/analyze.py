#!/usr/bin/env python3
"""Analyze the frozen 200-scene learned-Gate admission run."""

import argparse
import gzip
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
TOPOLOGY_KEYS = (
    "conflict_edge_count",
    "max_conflict_degree",
    "simultaneous_conflict_count",
)
RUN_NAME = "g3_learned_gate_n200_r1_s20260802_20260802_014005"


def parse_args():
    route_dir = Path(__file__).resolve().parents[1]
    main_dir = route_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate",
        type=Path,
        default=route_dir
        / "local_data/G3_learned_gate_validation/results"
        / f"{RUN_NAME}.npy",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=main_dir
        / "results/05_当前冻结方案"
        / "D4_dense_validation_actor_comparison_s20260728/5a_evaluation.npy",
    )
    parser.add_argument(
        "--oracle",
        type=Path,
        default=main_dir
        / "results/05_当前冻结方案"
        / "D4_dense_validation_actor_comparison_s20260728"
        / "5a_plus_epoch16_oracle_evaluation.npy",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=main_dir
        / "datasets/fixed_v1/views/dense_validation_monitor_v1/validation.json.gz",
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("summary.json")
    )
    parser.add_argument("--expected-episodes", type=int, default=200)
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
        raise ValueError(f"{label} is missing {len(missing)} evaluated scenario IDs")
    return np.asarray([row_by_id[scenario_id] for scenario_id in scenario_ids], dtype=object)


def load_topology(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        str(case["scenario_id"]): {
            key: int(case.get("metrics", {}).get(key, 0)) for key in TOPOLOGY_KEYS
        }
        for case in payload["scenarios"]
    }


def aggregate(rows, include_gate=False):
    episodes = len(rows)
    if not episodes:
        return None
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
        "full_success": float(
            rows[:, COLUMNS["full_success"]].astype(float).mean()
        ),
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


def exact_mcnemar_p(baseline_only, candidate_only):
    discordant = baseline_only + candidate_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k)
        for k in range(min(baseline_only, candidate_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def paired_comparison(reference, gate):
    reference_full = reference[:, COLUMNS["full_success"]].astype(bool)
    gate_full = gate[:, COLUMNS["full_success"]].astype(bool)
    reference_only = int(np.sum(reference_full & ~gate_full))
    gate_only = int(np.sum(~reference_full & gate_full))
    return {
        "both_success": int(np.sum(reference_full & gate_full)),
        "gate_only_success": gate_only,
        "reference_only_success": reference_only,
        "both_failure": int(np.sum(~reference_full & ~gate_full)),
        "mcnemar_exact_p": exact_mcnemar_p(reference_only, gate_only),
    }


def paired_bootstrap_delta(reference, gate, resamples, seed):
    rng = np.random.default_rng(seed)
    episode_delta = (
        gate[:, COLUMNS["full_success"]].astype(float)
        - reference[:, COLUMNS["full_success"]].astype(float)
    )
    indices = rng.integers(0, len(episode_delta), size=(resamples, len(episode_delta)))
    samples = episode_delta[indices].mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return {
        "metric": "full_success_delta",
        "resamples": resamples,
        "seed": seed,
        "mean": float(episode_delta.mean()),
        "ci95": [float(low), float(high)],
    }


def stratify(gate, baseline, oracle, topology, key):
    scenario_ids = [str(row[COLUMNS["scenario_id"]]) for row in gate]
    values = np.asarray([topology[scenario_id][key] for scenario_id in scenario_ids])
    result = {}
    for value in sorted(set(values.tolist())):
        mask = values == value
        result[str(value)] = {
            "scenarios": int(mask.sum()),
            "baseline": aggregate(baseline[mask]),
            "learned_gate": aggregate(gate[mask], include_gate=True),
            "oracle": aggregate(oracle[mask]),
        }
    return result


def subset_result(gate, baseline, oracle, topology, predicate):
    scenario_ids = [str(row[COLUMNS["scenario_id"]]) for row in gate]
    mask = np.asarray([predicate(topology[scenario_id]) for scenario_id in scenario_ids])
    return {
        "scenarios": int(mask.sum()),
        "baseline": aggregate(baseline[mask]),
        "learned_gate": aggregate(gate[mask], include_gate=True),
        "oracle": aggregate(oracle[mask]),
        "gate_vs_baseline": paired_comparison(baseline[mask], gate[mask]),
    }


def severe_cases(gate, baseline, oracle, topology):
    cases = []
    for gate_row, baseline_row, oracle_row in zip(gate, baseline, oracle):
        success = int(gate_row[COLUMNS["success"]])
        if success > 1:
            continue
        scenario_id = str(gate_row[COLUMNS["scenario_id"]])
        cases.append(
            {
                "episode": int(gate_row[COLUMNS["episode"]]),
                "scenario_id": scenario_id,
                "agent_success": success,
                "collision": int(gate_row[COLUMNS["collision"]]),
                "timeout": int(gate_row[COLUMNS["timeout"]]),
                "dense_action_share": float(
                    gate_row[COLUMNS["dense_action_share"]]
                ),
                "gate_switches": int(gate_row[COLUMNS["gate_switches"]]),
                "gate_mean_probability": float(
                    gate_row[COLUMNS["gate_mean_probability"]]
                ),
                "baseline_full_success": int(
                    baseline_row[COLUMNS["full_success"]]
                ),
                "oracle_full_success": int(oracle_row[COLUMNS["full_success"]]),
                **topology[scenario_id],
            }
        )
    return cases


def main():
    args = parse_args()
    gate = load_rows(args.gate, 17)
    if len(gate) != args.expected_episodes:
        raise ValueError(
            f"Gate run has {len(gate)} rows; expected {args.expected_episodes}. "
            "Do not analyze a partial admission run."
        )
    scenario_ids = [str(row[COLUMNS["scenario_id"]]) for row in gate]
    baseline = select_by_id(load_rows(args.baseline, 13), scenario_ids, "baseline")
    oracle = select_by_id(load_rows(args.oracle, 13), scenario_ids, "oracle")
    topology = load_topology(args.manifest)
    missing_topology = [item for item in scenario_ids if item not in topology]
    if missing_topology:
        raise ValueError(f"Manifest is missing {len(missing_topology)} scenario IDs")

    baseline_metrics = aggregate(baseline)
    gate_metrics = aggregate(gate, include_gate=True)
    oracle_metrics = aggregate(oracle)
    denominator = oracle_metrics["full_success"] - baseline_metrics["full_success"]
    recovery = (
        (gate_metrics["full_success"] - baseline_metrics["full_success"])
        / denominator
        if denominator > 0
        else None
    )
    gate_full = gate[:, COLUMNS["full_success"]].astype(bool)
    activation_diagnostics = {
        "full_success": aggregate(gate[gate_full], include_gate=True),
        "failure": aggregate(gate[~gate_full], include_gate=True),
    }

    edge_ge_2 = subset_result(
        gate, baseline, oracle, topology, lambda metrics: metrics["conflict_edge_count"] >= 2
    )
    max_degree_ge_2 = subset_result(
        gate, baseline, oracle, topology, lambda metrics: metrics["max_conflict_degree"] >= 2
    )
    paired = paired_comparison(baseline, gate)
    checks = {
        "full_success_or_recovery": bool(
            gate_metrics["full_success"] >= 0.45
            or (recovery is not None and recovery >= 0.60)
        ),
        "paired_improvements_exceed_regressions": bool(
            paired["gate_only_success"] > paired["reference_only_success"]
        ),
        "no_timeout_increase": bool(
            gate_metrics["timeout_count"] <= baseline_metrics["timeout_count"]
        ),
        "positive_edge_ge_2_gain": bool(
            edge_ge_2["learned_gate"]["full_success"]
            > edge_ge_2["baseline"]["full_success"]
        ),
        "positive_max_degree_ge_2_gain": bool(
            max_degree_ge_2["learned_gate"]["full_success"]
            > max_degree_ge_2["baseline"]["full_success"]
        ),
    }
    result = {
        "result_version": 1,
        "status": "complete",
        "protocol": {
            "run_name": RUN_NAME,
            "episodes": len(gate),
            "seed": 20260802,
            "switch_on_threshold": 0.44,
            "switch_off_threshold": 0.34,
            "minimum_hold_steps": 3,
            "gate_stats": str(args.gate),
            "historical_baseline_stats": str(args.baseline),
            "historical_oracle_stats": str(args.oracle),
            "manifest": str(args.manifest),
            "historical_comparison_limitation": (
                "Baseline and Oracle are same-ID development references from an older "
                "20260728 run, not final same-process paper statistics."
            ),
        },
        "baseline": baseline_metrics,
        "learned_gate": gate_metrics,
        "oracle": oracle_metrics,
        "oracle_full_success_gain_recovery": recovery,
        "gate_vs_baseline": paired,
        "gate_vs_oracle": paired_comparison(oracle, gate),
        "paired_bootstrap": paired_bootstrap_delta(
            baseline, gate, args.bootstrap_resamples, args.bootstrap_seed
        ),
        "activation_diagnostics": activation_diagnostics,
        "by_conflict_edge_count": stratify(
            gate, baseline, oracle, topology, "conflict_edge_count"
        ),
        "by_max_conflict_degree": stratify(
            gate, baseline, oracle, topology, "max_conflict_degree"
        ),
        "by_simultaneous_conflict_count": stratify(
            gate, baseline, oracle, topology, "simultaneous_conflict_count"
        ),
        "subsets": {
            "edge_ge_2": edge_ge_2,
            "edge_ge_3": subset_result(
                gate,
                baseline,
                oracle,
                topology,
                lambda metrics: metrics["conflict_edge_count"] >= 3,
            ),
            "max_degree_ge_2": max_degree_ge_2,
            "simultaneous_ge_2": subset_result(
                gate,
                baseline,
                oracle,
                topology,
                lambda metrics: metrics["simultaneous_conflict_count"] >= 2,
            ),
        },
        "severe_cases_agent_success_le_1": severe_cases(
            gate, baseline, oracle, topology
        ),
        "admission_checks": checks,
        "admission_pass": bool(all(checks.values())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
