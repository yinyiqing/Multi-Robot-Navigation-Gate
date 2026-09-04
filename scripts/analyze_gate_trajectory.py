#!/usr/bin/env python3
"""Summarize executed Actor-mode sequences in a Router trajectory JSONL."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


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


def compress_modes(step_mode_pairs):
    runs = []
    for step, mode in step_mode_pairs:
        if not runs or runs[-1]["mode"] != mode:
            runs.append({"mode": mode, "start_step": step, "end_step": step, "steps": 1})
        else:
            runs[-1]["end_step"] = step
            runs[-1]["steps"] += 1
    return runs


def cycle_records(seed, episode, scenario_id, robot, runs, full_success):
    cycles = []
    for index in range(len(runs) - 2):
        window = runs[index : index + 3]
        if [item["mode"] for item in window] != ["standard", "dense", "standard"]:
            continue
        pre, interaction, post = window
        cycles.append(
            {
                "seed": seed,
                "episode": episode,
                "scenario_id": scenario_id,
                "robot": robot,
                "full_success": full_success,
                "cycle_start_step": pre["start_step"],
                "interaction_start_step": interaction["start_step"],
                "interaction_end_step": interaction["end_step"],
                "cycle_end_step": post["end_step"],
                "pre_standard_steps": pre["steps"],
                "interaction_steps": interaction["steps"],
                "post_standard_steps": post["steps"],
                "minimum_phase_steps": min(pre["steps"], interaction["steps"], post["steps"]),
            }
        )
    return cycles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    episodes = load_episodes(args.trajectory)
    results = np.load(args.results, allow_pickle=True)
    if results.ndim != 2 or results.shape[1] != 17 or len(results) != len(episodes):
        raise SystemExit("trajectory/result count or schema mismatch")

    per_robot = []
    cycles = []
    for episode, row in enumerate(results, start=1):
        records = episodes.get(episode)
        scenario_id = str(row[12])
        if not records or str(records[0]["case"]) != scenario_id:
            raise SystemExit(f"trajectory/result order mismatch at episode {episode}")
        robots = sorted(records[0]["positions"])
        for robot_index, robot in enumerate(robots):
            active = []
            probabilities = []
            for record in records:
                mode = (record.get("actor_modes") or {}).get(robot)
                mask = record.get("active_before") or []
                if mode is None or robot_index >= len(mask) or not mask[robot_index]:
                    continue
                active.append((int(record["step"]), mode))
                probability = (record.get("gate_probabilities") or {}).get(robot)
                if isinstance(probability, (int, float)) and np.isfinite(probability):
                    probabilities.append(float(probability))
            runs = compress_modes(active)
            signature = " -> ".join(item["mode"] for item in runs)
            visible_switches = max(len(runs) - 1, 0)
            controller_switches = visible_switches + int(bool(runs) and runs[0]["mode"] == "dense")
            per_robot.append(
                {
                    "seed": args.seed,
                    "episode": episode,
                    "scenario_id": scenario_id,
                    "robot": robot,
                    "full_success": int(row[8]),
                    "active_steps": len(active),
                    "first_mode": runs[0]["mode"] if runs else None,
                    "last_mode": runs[-1]["mode"] if runs else None,
                    "first_probability": probabilities[0] if probabilities else None,
                    "mean_probability": float(np.mean(probabilities)) if probabilities else None,
                    "visible_switches": visible_switches,
                    "controller_switches": controller_switches,
                    "run_signature": signature,
                    "run_lengths": ";".join(f"{item['mode']}:{item['steps']}" for item in runs),
                }
            )
            cycles.extend(
                cycle_records(
                    args.seed,
                    episode,
                    scenario_id,
                    robot,
                    runs,
                    int(row[8]),
                )
            )

    reconstructed = defaultdict(int)
    for item in per_robot:
        reconstructed[item["episode"]] += item["controller_switches"]
    mismatches = []
    for episode, row in enumerate(results, start=1):
        if reconstructed[episode] != int(row[15]):
            mismatches.append(
                {"episode": episode, "recorded": int(row[15]), "reconstructed": reconstructed[episode]}
            )
    cycles.sort(
        key=lambda item: (
            item["full_success"],
            item["minimum_phase_steps"],
            item["pre_standard_steps"] + item["interaction_steps"] + item["post_standard_steps"],
        ),
        reverse=True,
    )
    signatures = Counter(item["run_signature"] for item in per_robot)
    summary = {
        "trajectory": str(args.trajectory),
        "results": str(args.results),
        "episodes": len(episodes),
        "robot_sequences": len(per_robot),
        "controller_switch_reconstruction_mismatches": mismatches,
        "first_mode_counts": dict(Counter(item["first_mode"] for item in per_robot)),
        "signature_counts": dict(signatures.most_common()),
        "complete_standard_dense_standard_cycles": len(cycles),
        "robots_with_complete_cycle": len({(item["episode"], item["robot"]) for item in cycles}),
        "ranked_cycle_candidates": cycles,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "mode_sequence_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "mode_sequences.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_robot[0]))
        writer.writeheader()
        writer.writerows(per_robot)
    with (args.output_dir / "complete_cycles.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(cycles[0]) if cycles else [
            "seed", "episode", "scenario_id", "robot", "full_success",
            "cycle_start_step", "interaction_start_step", "interaction_end_step",
            "cycle_end_step", "pre_standard_steps", "interaction_steps",
            "post_standard_steps", "minimum_phase_steps",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cycles)
    print(args.output_dir / "mode_sequence_summary.json")
    print(f"complete cycles: {len(cycles)} across {summary['robots_with_complete_cycle']} robots")


if __name__ == "__main__":
    main()
