#!/usr/bin/env python3
import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from scenario_manifests import GridPlanner, load_manifest_dataset


def position_on_path(path, distance):
    remaining = max(float(distance), 0.0)
    for start, end in zip(path, path[1:]):
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        length = float(np.linalg.norm(end - start))
        if length <= 1e-9:
            continue
        if remaining <= length:
            return start + remaining / length * (end - start)
        remaining -= length
    return np.asarray(path[-1], dtype=float)


def path_length(path):
    return sum(math.dist(start, end) for start, end in zip(path, path[1:]))


def delayed_trajectories(path, delays, speed, sample_dt, horizon):
    times = np.arange(0.0, horizon + 0.5 * sample_dt, sample_dt)
    trajectories = []
    for delay in delays:
        traveled = np.maximum(times - float(delay), 0.0) * speed
        trajectories.append(
            np.asarray([position_on_path(path, distance) for distance in traveled])
        )
    return np.asarray(trajectories)


def pair_compatibility(left_trajectories, right_trajectories, separation):
    differences = (
        left_trajectories[:, None, :, :] - right_trajectories[None, :, :, :]
    )
    min_squared_distance = np.min(np.sum(differences * differences, axis=-1), axis=-1)
    return min_squared_distance >= float(separation) ** 2


def find_start_delay_schedule(
    paths,
    speed=0.5,
    separation=0.7,
    delay_step=0.5,
    max_delay=8.0,
    sample_dt=0.1,
):
    names = tuple(sorted(paths))
    delays = np.arange(0.0, max_delay + 0.5 * delay_step, delay_step)
    durations = {name: path_length(paths[name]) / speed for name in names}
    horizon = max_delay + max(durations.values(), default=0.0) + sample_dt
    trajectories = {
        name: delayed_trajectories(paths[name], delays, speed, sample_dt, horizon)
        for name in names
    }
    compatibility = {}
    degrees = Counter()
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            matrix = pair_compatibility(
                trajectories[left], trajectories[right], separation
            )
            compatibility[(left, right)] = matrix
            if not bool(matrix[0, 0]):
                degrees[left] += 1
                degrees[right] += 1

    def compatible(name, delay_index, assigned):
        for other, other_delay_index in assigned.items():
            if name < other:
                allowed = compatibility[(name, other)][delay_index, other_delay_index]
            else:
                allowed = compatibility[(other, name)][other_delay_index, delay_index]
            if not bool(allowed):
                return False
        return True

    explored_nodes = 0
    for max_delay_index in range(len(delays)):
        candidates = range(max_delay_index + 1)
        for anchor in names:
            assigned = {anchor: 0}
            order = sorted(
                (name for name in names if name != anchor),
                key=lambda name: (-degrees[name], name),
            )

            def search(position):
                nonlocal explored_nodes
                explored_nodes += 1
                if position == len(order):
                    return dict(assigned)
                name = order[position]
                for delay_index in candidates:
                    if not compatible(name, delay_index, assigned):
                        continue
                    assigned[name] = delay_index
                    result = search(position + 1)
                    if result is not None:
                        return result
                    assigned.pop(name)
                return None

            schedule = search(0)
            if schedule is not None:
                schedule_seconds = {
                    name: round(float(delays[index]), 6)
                    for name, index in sorted(schedule.items())
                }
                return {
                    "solved": True,
                    "delays_s": schedule_seconds,
                    "max_delay_s": max(schedule_seconds.values(), default=0.0),
                    "explored_nodes": explored_nodes,
                }
    return {
        "solved": False,
        "delays_s": None,
        "max_delay_s": None,
        "explored_nodes": explored_nodes,
    }


def scenario_paths(scenario, planner):
    boxes = [tuple(map(float, box)) for box in scenario.get("boxes", [])]
    paths = {}
    for name, agent in sorted(scenario["agents"].items()):
        path = planner.plan(agent["start"], agent["goal"], boxes)
        if path is None:
            return None
        paths[name] = path
    return paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check whether fixed paths become collision-free by delaying starts."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-scenarios", type=int, default=0)
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--separation", type=float, default=0.7)
    parser.add_argument("--delay-step", type=float, default=0.5)
    parser.add_argument("--max-delay", type=float, default=8.0)
    parser.add_argument("--sample-dt", type=float, default=0.1)
    parser.add_argument("--planner-clearance", type=float, default=0.24)
    return parser.parse_args()


def main():
    args = parse_args()
    if any(
        value <= 0.0
        for value in (
            args.speed,
            args.separation,
            args.delay_step,
            args.sample_dt,
            args.planner_clearance,
        )
    ) or args.max_delay < 0.0:
        raise ValueError("Probe distances, speeds, and time increments must be positive")

    payload = load_manifest_dataset(args.manifest)
    scenarios = payload["scenarios"]
    if args.max_scenarios > 0:
        scenarios = scenarios[: args.max_scenarios]
    planner = GridPlanner(0.15, args.planner_clearance)
    results = []
    for index, scenario in enumerate(scenarios, start=1):
        paths = scenario_paths(scenario, planner)
        if paths is None:
            result = {"solved": False, "reason": "no_static_path"}
        else:
            result = find_start_delay_schedule(
                paths,
                speed=args.speed,
                separation=args.separation,
                delay_step=args.delay_step,
                max_delay=args.max_delay,
                sample_dt=args.sample_dt,
            )
            result["reason"] = "scheduled" if result["solved"] else "no_delay_schedule"
        record = {
            "scenario_id": scenario["scenario_id"],
            "conflict_edge_count": scenario.get("metrics", {}).get(
                "conflict_edge_count"
            ),
            **result,
        }
        results.append(record)
        if index % 25 == 0 or index == len(scenarios):
            solved = sum(item["solved"] for item in results)
            print("processed=%i/%i solved=%i" % (index, len(scenarios), solved))

    solved = [item for item in results if item["solved"]]
    by_edges = defaultdict(lambda: {"total": 0, "solved": 0})
    for item in results:
        key = str(item["conflict_edge_count"])
        by_edges[key]["total"] += 1
        by_edges[key]["solved"] += int(item["solved"])
    summary = {
        "manifest": str(args.manifest),
        "scenario_count": len(results),
        "solved_count": len(solved),
        "solved_rate": len(solved) / len(results) if results else 0.0,
        "parameters": {
            "speed_mps": args.speed,
            "separation_m": args.separation,
            "delay_step_s": args.delay_step,
            "max_delay_s": args.max_delay,
            "sample_dt_s": args.sample_dt,
            "planner_clearance_m": args.planner_clearance,
        },
        "solved_by_conflict_edges": dict(sorted(by_edges.items())),
        "mean_required_max_delay_s": (
            float(np.mean([item["max_delay_s"] for item in solved])) if solved else None
        ),
        "results": results,
    }
    public_summary = {key: value for key, value in summary.items() if key != "results"}
    print(json.dumps(public_summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
