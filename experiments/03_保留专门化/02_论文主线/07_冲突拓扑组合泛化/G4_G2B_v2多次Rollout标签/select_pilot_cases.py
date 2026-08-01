#!/usr/bin/env python3
"""Build a deterministic G2-B v2 label-stability pilot manifest."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


FULL_SUCCESS_COLUMN = 8
SCENARIO_ID_COLUMN = 12
CATEGORY_PREDICATES = {
    "gate_improved": lambda baseline, gate, oracle: not baseline and gate,
    "gate_regressed": lambda baseline, gate, oracle: baseline and not gate,
    "oracle_rescue_gap": lambda baseline, gate, oracle: (
        not baseline and not gate and oracle
    ),
}


def parse_args():
    route_dir = Path(__file__).resolve().parents[1]
    main_dir = route_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate",
        type=Path,
        default=route_dir
        / "local_data/G3_learned_gate_validation/results"
        / "g3_learned_gate_n200_r1_s20260802_20260802_014005.npy",
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
        "--output-manifest",
        type=Path,
        default=Path(__file__).with_name("pilot_manifest.json"),
    )
    parser.add_argument(
        "--output-selection",
        type=Path,
        default=Path(__file__).with_name("pilot_selection.json"),
    )
    parser.add_argument("--cases-per-category", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


def load_json(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def rows_by_id(path):
    rows = np.load(path, allow_pickle=True)
    if rows.ndim != 2 or rows.shape[1] < 13:
        raise ValueError(f"Unexpected result shape for {path}: {rows.shape}")
    result = {str(row[SCENARIO_ID_COLUMN]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate scenario IDs in {path}")
    return result


def edge_band(edge_count):
    if edge_count <= 1:
        return "1"
    if edge_count == 2:
        return "2"
    return "3+"


def deterministic_rank(scenario_id, seed):
    value = f"{seed}:{scenario_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def select_category(candidates, count, seed):
    preferred_bands = ("1", "2", "3+")
    selected = []
    for band in preferred_bands:
        if len(selected) >= count:
            break
        matching = [item for item in candidates if item["edge_band"] == band]
        if matching:
            selected.append(
                min(
                    matching,
                    key=lambda item: deterministic_rank(item["scenario_id"], seed),
                )
            )
    remaining = [item for item in candidates if item not in selected]
    remaining.sort(key=lambda item: deterministic_rank(item["scenario_id"], seed))
    selected.extend(remaining[: max(0, count - len(selected))])
    return selected


def main():
    args = parse_args()
    if args.cases_per_category < 1:
        raise ValueError("cases-per-category must be positive")
    manifest = load_json(args.manifest)
    gate = rows_by_id(args.gate)
    baseline = rows_by_id(args.baseline)
    oracle = rows_by_id(args.oracle)
    manifest_cases = {
        str(case["scenario_id"]): case for case in manifest["scenarios"]
    }

    candidate_groups = {category: [] for category in CATEGORY_PREDICATES}
    for scenario_id, gate_row in gate.items():
        if scenario_id not in baseline or scenario_id not in oracle:
            raise ValueError(f"Historical results are missing scenario {scenario_id}")
        if scenario_id not in manifest_cases:
            raise ValueError(f"Manifest is missing scenario {scenario_id}")
        baseline_full = bool(baseline[scenario_id][FULL_SUCCESS_COLUMN])
        gate_full = bool(gate_row[FULL_SUCCESS_COLUMN])
        oracle_full = bool(oracle[scenario_id][FULL_SUCCESS_COLUMN])
        metrics = manifest_cases[scenario_id].get("metrics", {})
        record = {
            "scenario_id": scenario_id,
            "conflict_edge_count": int(metrics.get("conflict_edge_count", 0)),
            "max_conflict_degree": int(metrics.get("max_conflict_degree", 0)),
            "simultaneous_conflict_count": int(
                metrics.get("simultaneous_conflict_count", 0)
            ),
            "edge_band": edge_band(int(metrics.get("conflict_edge_count", 0))),
            "baseline_full_success": int(baseline_full),
            "gate_full_success": int(gate_full),
            "oracle_full_success": int(oracle_full),
        }
        for category, predicate in CATEGORY_PREDICATES.items():
            if predicate(baseline_full, gate_full, oracle_full):
                candidate_groups[category].append(record)

    selected = []
    counts = {}
    for category, candidates in candidate_groups.items():
        category_selected = select_category(
            candidates, args.cases_per_category, args.seed
        )
        if len(category_selected) < args.cases_per_category:
            raise ValueError(
                f"Category {category} has only {len(category_selected)} selectable cases"
            )
        counts[category] = len(candidates)
        for item in category_selected:
            selected.append({"category": category, **item})

    selected_ids = [item["scenario_id"] for item in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Pilot categories unexpectedly selected duplicate scenarios")
    selected_cases = [manifest_cases[scenario_id] for scenario_id in selected_ids]
    output_manifest = dict(manifest)
    output_manifest.update(
        {
            "dataset_id": "g4_g2b_v2_label_stability_pilot",
            "split": "diagnostic_pilot",
            "view_config": {
                "purpose": "counterfactual_label_stability_only",
                "source_run": (
                    "g3_learned_gate_n200_r1_s20260802_20260802_014005"
                ),
                "selection_seed": args.seed,
                "cases_per_category": args.cases_per_category,
                "prohibit_performance_evaluation": True,
            },
            "scenarios": selected_cases,
        }
    )
    selection = {
        "selection_version": 1,
        "seed": args.seed,
        "cases_per_category": args.cases_per_category,
        "candidate_counts": counts,
        "selected": selected,
        "usage_restriction": (
            "Diagnostic label-stability pilot only. Do not train and evaluate a Gate "
            "on these same admission scenarios."
        ),
    }
    args.output_manifest.write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_selection.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(selection, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
