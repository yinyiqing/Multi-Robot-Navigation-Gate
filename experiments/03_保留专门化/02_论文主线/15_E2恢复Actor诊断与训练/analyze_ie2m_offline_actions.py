#!/usr/bin/env python3
import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "TD3"))
from actor_models import Actor


AGENTS = ("r1", "r2", "r3", "r4", "r5")


def load_actor(path):
    actor = Actor(24, 2)
    actor.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    actor.eval()
    return actor


def infer(actor, states):
    if not states:
        return np.empty((0, 2), dtype=np.float32)
    with torch.no_grad():
        actions = actor(torch.as_tensor(np.asarray(states), dtype=torch.float32)).numpy()
    actions[:, 0] = (actions[:, 0] + 1.0) / 2.0
    return actions


def summarize(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--actor", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    with gzip.open(args.manifest, "rt", encoding="utf-8") as handle:
        scenarios = json.load(handle)["scenarios"]
    metadata = {
        str(item["scenario_id"]): str(item["view"]["capacity_topology"])
        for item in scenarios
    }
    results = np.load(args.result, allow_pickle=True)
    outcome = {
        str(row[12]): {
            "full_success": int(row[8]),
            "collision": int(row[7]),
            "unresolved": int(row[10]),
            "timeout": int(row[11]),
        }
        for row in results
    }
    actors = {name: load_actor(Path(path)) for name, path in args.actor}

    rows = []
    states = []
    with args.trajectory.open("r", encoding="utf-8") as handle:
        for line in handle:
            frame = json.loads(line)
            scenario_id = str(frame["case"])
            modes = frame.get("actor_modes") or {}
            for index, name in enumerate(AGENTS):
                if not bool(frame["active_before"][index]) or modes.get(name) != "dense":
                    continue
                info = frame["agents"][name]
                rows.append({
                    "scenario_id": scenario_id,
                    "topology": metadata[scenario_id],
                    "episode_full_success": outcome[scenario_id]["full_success"],
                    "episode_collision": outcome[scenario_id]["collision"],
                    "episode_unresolved": outcome[scenario_id]["unresolved"],
                    "episode_timeout": outcome[scenario_id]["timeout"],
                    "step": int(frame["step"]),
                    "agent": name,
                    "progress": float(info["progress"]),
                    "nearest_robot_distance": info.get("nearest_robot_distance"),
                    "executed_linear": float(frame["actions"][index][0]),
                    "executed_angular": float(frame["actions"][index][1]),
                })
                states.append(frame["actor_states"][index])

    predictions = {name: infer(actor, states) for name, actor in actors.items()}
    for index, row in enumerate(rows):
        for name, actions in predictions.items():
            row[f"{name}_linear"] = float(actions[index, 0])
            row[f"{name}_angular"] = float(actions[index, 1])

    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups["all"].append(index)
        groups[row["topology"]].append(index)
        groups["failed" if not row["episode_full_success"] else "success"].append(index)
        if row["episode_collision"]:
            groups["collision_episode"].append(index)
    summary = {}
    for group, indices in groups.items():
        item = {"frames": len(indices)}
        for actor_name in actors:
            linear = predictions[actor_name][indices, 0]
            angular = np.abs(predictions[actor_name][indices, 1])
            item[actor_name] = {
                "linear": summarize(linear),
                "abs_angular": summarize(angular),
                "near_stop_rate": float(np.mean(linear <= 0.05)),
                "high_speed_rate": float(np.mean(linear >= 0.6)),
            }
        summary[group] = item

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "audit": {
            "trajectory_interaction_frames": len(rows),
            "actors": {name: str(path) for name, path in args.actor},
        },
        "groups": summary,
    }
    (args.output_dir / "offline_actions.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "offline_actions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
