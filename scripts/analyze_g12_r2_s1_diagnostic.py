#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = PROJECT_ROOT / "experiments/02_课程学习/cases"
RUN_DIR = (
    PROJECT_ROOT
    / "experiments/03_保留专门化/02_论文主线"
    / "12_参数匹配单Actor容量对照/local_data/s1_diagnostic"
)
REPEATS = 3
STAGES = {
    "stage1_single": "stage1_single_local_cases.json",
    "stage1e_single_rescue": "stage1e_single_rescue_cases.json",
    "stage1f_wall_parallel_rescue": "stage1f_wall_parallel_rescue_cases.json",
    "stage1g_collision_guard": "stage1g_collision_guard_cases.json",
}
EXPECTED_CASE_SHA256 = {
    "stage1_single": "9cc79ec2a82908127c77fa00eff1448661814f8025176d769ccd4a03a8fb4b40",
    "stage1e_single_rescue": "3b2566d8898d5380bc4d5295009d0b81e088bd96bec63939d5184d88a8cce4d9",
    "stage1f_wall_parallel_rescue": "36906f6164d79551a09264f10e939779c68b8ca8ab366e78b50911f73974f563",
    "stage1g_collision_guard": "d52dd8d1b5dd904ad7f4b8c55b60a258fc5cc4616469ead9932d93ee11be4403",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def case_status(successes):
    if successes == REPEATS:
        return "pass"
    if successes == REPEATS - 1:
        return "borderline"
    return "repair"


def analyze_stage(stage, results_dir):
    case_path = CASE_DIR / STAGES[stage]
    actual_hash = sha256(case_path)
    if actual_hash != EXPECTED_CASE_SHA256[stage]:
        raise ValueError(f"{stage} case hash mismatch: {actual_hash}")
    with case_path.open("r", encoding="utf-8") as handle:
        case_names = [str(item["name"]) for item in json.load(handle)["cases"]]

    result_path = results_dir / f"{stage}.npy"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    rows = np.load(result_path, allow_pickle=True)
    expected_rows = len(case_names) * REPEATS
    if rows.shape != (expected_rows, 17):
        raise ValueError(f"{stage} result shape is {rows.shape}, expected {(expected_rows, 17)}")

    observed = [str(value) for value in rows[:, 12]]
    expected_order = case_names * REPEATS
    if observed != expected_order:
        raise ValueError(f"{stage} case order or repeat coverage does not match the protocol")

    cases = {}
    for name in case_names:
        mask = np.asarray(observed) == name
        selected = rows[mask]
        successes = int(np.asarray(selected[:, 8], dtype=int).sum())
        collisions = int(np.asarray(selected[:, 7], dtype=int).sum())
        unresolved = int(np.asarray(selected[:, 10], dtype=int).sum())
        timeouts = int(np.asarray(selected[:, 11], dtype=int).sum())
        if successes + collisions + unresolved != REPEATS:
            raise ValueError(f"{stage}/{name} outcomes do not sum to {REPEATS}")
        cases[name] = {
            "successes": successes,
            "collisions": collisions,
            "unresolved": unresolved,
            "timeouts": timeouts,
            "status": case_status(successes),
            "mean_steps": float(np.asarray(selected[:, 3], dtype=float).mean()),
            "mean_final_distance": float(np.asarray(selected[:, 9], dtype=float).mean()),
        }

    statuses = [item["status"] for item in cases.values()]
    stage_status = "repair" if "repair" in statuses else (
        "borderline" if "borderline" in statuses else "pass"
    )
    total_success = sum(item["successes"] for item in cases.values())
    return {
        "stage": stage,
        "status": stage_status,
        "episodes": expected_rows,
        "full_success_rate": total_success / expected_rows,
        "case_sha256": actual_hash,
        "result_sha256": sha256(result_path),
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(STAGES))
    parser.add_argument("--results-dir", type=Path, default=RUN_DIR / "results")
    parser.add_argument("--output", type=Path, default=RUN_DIR / "summary.json")
    args = parser.parse_args()

    selected = [args.stage] if args.stage else list(STAGES)
    stages = {stage: analyze_stage(stage, args.results_dir) for stage in selected}
    statuses = [item["status"] for item in stages.values()]
    overall = "repair" if "repair" in statuses else (
        "borderline" if "borderline" in statuses else "pass"
    )
    payload = {
        "protocol": "g12-r2-s1-diagnostic-v1",
        "evaluation_seed": 20260812,
        "repeats_per_case": REPEATS,
        "overall_status": overall,
        "stages": stages,
    }
    if args.stage:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
