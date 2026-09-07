#!/usr/bin/env python3
"""Audit per-robot mode cycles in the post-sealed high-switch capture."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROBOT_NAMES = tuple(f"r{index}" for index in range(1, 6))


def compressed_runs(modes):
    runs = []
    for mode in modes:
        if not runs or runs[-1][0] != mode:
            runs.append([mode, 1])
        else:
            runs[-1][1] += 1
    return [(mode, length) for mode, length in runs]


def contains_pattern(runs, pattern):
    values = [mode for mode, _ in runs]
    width = len(pattern)
    return any(tuple(values[index : index + width]) == pattern for index in range(len(values) - width + 1))


def load_episodes(path):
    episodes = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                episodes[int(record["episode"])].append(record)
    for records in episodes.values():
        records.sort(key=lambda item: int(item["step"]))
    return dict(episodes)


def finite_float(value):
    if isinstance(value, (int, float)) and np.isfinite(value):
        return float(value)
    return None


def summarize_robot(seed, scenario_id, robot, records):
    active_records = []
    robot_index = ROBOT_NAMES.index(robot)
    for record in records:
        modes = record.get("actor_modes") or {}
        active_before = record.get("active_before") or []
        if robot in modes and robot_index < len(active_before) and active_before[robot_index]:
            active_records.append(record)
    modes = [(record.get("actor_modes") or {})[robot] for record in active_records]
    probabilities = [
        finite_float((record.get("gate_probabilities") or {}).get(robot))
        for record in active_records
    ]
    runs = compressed_runs(modes)
    visible_switches = max(len(runs) - 1, 0)
    # GateHysteresis starts in standard. If the first executed action is dense,
    # its initial standard->dense transition is included in episode_stats().
    controller_switches = visible_switches + int(bool(runs) and runs[0][0] == "dense")
    return {
        "seed": seed,
        "scenario_id": scenario_id,
        "robot": robot,
        "active_steps": len(modes),
        "first_mode": modes[0] if modes else None,
        "last_mode": modes[-1] if modes else None,
        "first_probability": next((value for value in probabilities if value is not None), None),
        "mean_probability": float(np.mean([value for value in probabilities if value is not None])) if any(value is not None for value in probabilities) else None,
        "standard_steps": modes.count("standard"),
        "interaction_steps": modes.count("dense"),
        "visible_switches": visible_switches,
        "controller_switches": controller_switches,
        "executed_standard_dense_standard": contains_pattern(runs, ("standard", "dense", "standard")),
        "executed_dense_standard_dense": contains_pattern(runs, ("dense", "standard", "dense")),
        "run_signature": " -> ".join(mode for mode, _ in runs),
        "run_lengths": ";".join(f"{mode}:{length}" for mode, length in runs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    selection_path = args.root / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    per_robot = []
    per_episode = []
    for group in selection["groups"]:
        seed = int(group["seed"])
        trajectory_path = args.root / "trajectories" / f"gate_cycle_capture_b2_s{seed}.jsonl"
        result_path = args.root / "results" / f"gate_cycle_capture_b2_s{seed}.npy"
        episodes = load_episodes(trajectory_path)
        rows = np.load(result_path, allow_pickle=True)
        if len(rows) != int(group["episodes"]) or len(episodes) != int(group["episodes"]):
            raise SystemExit(f"capture count mismatch for seed {seed}")
        sealed_by_id = {item["scenario_id"]: item for item in group["rows"]}
        for episode_index, row in enumerate(rows, start=1):
            scenario_id = str(row[12])
            records = episodes.get(episode_index)
            if not records or str(records[0]["case"]) != scenario_id:
                raise SystemExit(f"trajectory/result order mismatch for seed {seed}, episode {episode_index}")
            robot_rows = [
                summarize_robot(seed, scenario_id, robot, records) for robot in ROBOT_NAMES
            ]
            per_robot.extend(robot_rows)
            sealed = sealed_by_id[scenario_id]
            per_episode.append(
                {
                    "seed": seed,
                    "scenario_id": scenario_id,
                    "sealed_gate_switches": int(sealed["sealed_gate_switches"]),
                    "capture_gate_switches": int(row[15]),
                    "reconstructed_controller_switches": sum(item["controller_switches"] for item in robot_rows),
                    "capture_steps": int(row[3]),
                    "capture_full_success": int(row[8]),
                    "capture_interaction_share": float(row[14]),
                    "robots_with_standard_dense_standard": sum(item["executed_standard_dense_standard"] for item in robot_rows),
                    "robots_with_dense_standard_dense": sum(item["executed_dense_standard_dense"] for item in robot_rows),
                }
            )

    output_csv = args.root / "per_robot_cycles.csv"
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_robot[0]))
        writer.writeheader()
        writer.writerows(per_robot)

    signatures = Counter(item["run_signature"] for item in per_robot)
    first_probabilities = [item["first_probability"] for item in per_robot if item["first_probability"] is not None]
    interaction_run_lengths = []
    standard_run_lengths = []
    for item in per_robot:
        for part in item["run_lengths"].split(";"):
            if not part:
                continue
            mode, length = part.split(":")
            (interaction_run_lengths if mode == "dense" else standard_run_lengths).append(int(length))

    summary = {
        "purpose": selection["purpose"],
        "interpretation_scope": "qualitative post-sealed diagnosis; not a new estimate of G25 performance",
        "scene_repeat_pairs": len(per_episode),
        "robot_sequences": len(per_robot),
        "episodes": per_episode,
        "executed_cycle_counts": {
            "standard_dense_standard_robots": sum(item["executed_standard_dense_standard"] for item in per_robot),
            "dense_standard_dense_robots": sum(item["executed_dense_standard_dense"] for item in per_robot),
        },
        "first_mode_counts": dict(Counter(item["first_mode"] for item in per_robot)),
        "signature_counts": dict(signatures.most_common()),
        "first_probability": {
            "median": float(np.median(first_probabilities)),
            "minimum": float(np.min(first_probabilities)),
            "maximum": float(np.max(first_probabilities)),
            "at_or_above_switch_on_0_43": sum(value >= 0.43 for value in first_probabilities),
            "count": len(first_probabilities),
        },
        "run_length_steps": {
            "interaction_median": float(np.median(interaction_run_lengths)) if interaction_run_lengths else None,
            "interaction_maximum": max(interaction_run_lengths) if interaction_run_lengths else None,
            "standard_median": float(np.median(standard_run_lengths)) if standard_run_lengths else None,
            "standard_maximum": max(standard_run_lengths) if standard_run_lengths else None,
        },
    }
    summary_path = args.root / "cycle_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    readme = [
        "# High-switch Gate cycle capture",
        "",
        "This is a post-sealed qualitative diagnosis. It replays the nine G25 B2 scene/repeat rows",
        "whose original episode-level Gate switch count exceeded ten. It does not modify G25 statistics",
        "and must not be used as an unbiased performance estimate.",
        "",
        f"- Scene/repeat pairs: {len(per_episode)}",
        f"- Robot mode sequences: {len(per_robot)}",
        f"- Executed standard -> dense -> standard sequences: {summary['executed_cycle_counts']['standard_dense_standard_robots']}",
        f"- Executed dense -> standard -> dense sequences: {summary['executed_cycle_counts']['dense_standard_dense_robots']}",
        f"- First executed mode counts: {summary['first_mode_counts']}",
        "",
        "See `cycle_summary.json` for episode-level checks and `per_robot_cycles.csv` for the full audit.",
    ]
    (args.root / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(summary_path)
    print(output_csv)
    print(json.dumps(summary["executed_cycle_counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
