#!/usr/bin/env python3
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


AGENTS = ("r1", "r2", "r3", "r4", "r5")


def load_manifest(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        scenarios = json.load(handle)["scenarios"]
    return {
        str(item["scenario_id"]): {
            "pool": str(item["view"]["capacity_pool"]),
            "topology": str(item["view"]["capacity_topology"]),
            "conflict_edge_count": int(
                item["view"]["source_five_agent_conflict_edge_count"]
            ),
        }
        for item in scenarios
    }


def load_results(path, expected_ids):
    rows = np.load(path, allow_pickle=True)
    if rows.shape != (len(expected_ids), 17):
        raise ValueError(f"invalid result shape for {path}: {rows.shape}")
    observed = [str(row[12]) for row in rows]
    if observed != expected_ids or len(set(observed)) != len(observed):
        raise ValueError(f"result order or uniqueness audit failed for {path}")
    return {
        str(row[12]): {
            "full_success": int(row[8]),
            "success": int(row[6]),
            "collision": int(row[7]),
            "unresolved": int(row[10]),
            "timeout": int(row[11]),
            "steps": int(row[3]),
            "interaction_share": float(row[14]),
        }
        for row in rows
    }


def new_episode_stats():
    return {
        "frames": 0,
        "active_agent_frames": 0,
        "interaction_agent_frames": 0,
        "interaction_segments": 0,
        "interaction_linear": [],
        "interaction_abs_angular": [],
        "interaction_progress": [],
        "interaction_nearest": [],
        "interaction_local_degree": [],
        "interaction_high_speed_close": 0,
        "interaction_negative_progress": 0,
        "interaction_stagnant": 0,
        "interaction_multi_local": 0,
        "max_concurrent_interaction": 0,
        "collision_pre10_interaction": 0,
        "collision_pre10_linear": [],
    }


def summarize_values(values):
    if not values:
        return {"mean": None, "median": None, "p90": None}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
    }


def parse_trajectory(path):
    episodes = defaultdict(new_episode_stats)
    previous_modes = defaultdict(lambda: "standard")
    recent = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            frame = json.loads(line)
            scenario_id = str(frame["case"])
            episode = int(frame["episode"])
            key = (scenario_id, episode)
            stats = episodes[key]
            stats["frames"] += 1
            modes = frame.get("actor_modes") or {}
            agents = frame["agents"]
            actions = frame["actions"]
            positions = frame["positions"]
            active_before = frame["active_before"]
            concurrent = sum(
                bool(active_before[index]) and modes.get(name) == "dense"
                for index, name in enumerate(AGENTS)
            )
            stats["max_concurrent_interaction"] = max(
                stats["max_concurrent_interaction"], concurrent
            )
            for index, name in enumerate(AGENTS):
                if not bool(active_before[index]):
                    continue
                stats["active_agent_frames"] += 1
                mode = modes.get(name, "standard")
                if mode == "dense" and previous_modes[(key, name)] != "dense":
                    stats["interaction_segments"] += 1
                previous_modes[(key, name)] = mode

                action = actions[index]
                info = agents[name]
                origin = np.asarray(positions[name], dtype=float)
                nearby = 0
                for other_index, other in enumerate(AGENTS):
                    if other == name or not bool(active_before[other_index]):
                        continue
                    distance = float(
                        np.linalg.norm(np.asarray(positions[other], dtype=float) - origin)
                    )
                    if distance <= 2.0:
                        nearby += 1
                recent_key = (key, name)
                recent[recent_key].append(
                    {
                        "mode": mode,
                        "linear": float(action[0]),
                        "collision": bool(info["collision"]),
                    }
                )
                if len(recent[recent_key]) > 10:
                    del recent[recent_key][0]

                if mode != "dense":
                    continue
                stats["interaction_agent_frames"] += 1
                linear = float(action[0])
                progress = float(info["progress"])
                nearest = info.get("nearest_robot_distance")
                if nearest is not None and np.isfinite(float(nearest)):
                    nearest = float(nearest)
                    stats["interaction_nearest"].append(nearest)
                    if nearest <= 1.2 and linear >= 0.6:
                        stats["interaction_high_speed_close"] += 1
                stats["interaction_linear"].append(linear)
                stats["interaction_abs_angular"].append(abs(float(action[1])))
                stats["interaction_progress"].append(progress)
                stats["interaction_local_degree"].append(nearby)
                stats["interaction_negative_progress"] += progress < 0.0
                stats["interaction_stagnant"] += progress <= 0.003
                stats["interaction_multi_local"] += nearby >= 2

            for index, name in enumerate(AGENTS):
                if bool(agents[name]["collision"]):
                    window = recent[(key, name)]
                    stats["collision_pre10_interaction"] += sum(
                        item["mode"] == "dense" for item in window
                    )
                    stats["collision_pre10_linear"].extend(
                        item["linear"]
                        for item in window
                        if item["mode"] == "dense"
                    )
    by_scenario = {}
    for (scenario_id, _), stats in episodes.items():
        interaction_frames = stats["interaction_agent_frames"]
        by_scenario[scenario_id] = {
            "frames": stats["frames"],
            "active_agent_frames": stats["active_agent_frames"],
            "interaction_agent_frames": interaction_frames,
            "interaction_share": interaction_frames
            / max(stats["active_agent_frames"], 1),
            "interaction_segments": stats["interaction_segments"],
            "interaction_linear": summarize_values(stats["interaction_linear"]),
            "interaction_abs_angular": summarize_values(
                stats["interaction_abs_angular"]
            ),
            "interaction_progress": summarize_values(stats["interaction_progress"]),
            "interaction_nearest": summarize_values(stats["interaction_nearest"]),
            "interaction_local_degree": summarize_values(
                stats["interaction_local_degree"]
            ),
            "high_speed_close_rate": stats["interaction_high_speed_close"]
            / max(interaction_frames, 1),
            "negative_progress_rate": stats["interaction_negative_progress"]
            / max(interaction_frames, 1),
            "stagnant_rate": stats["interaction_stagnant"]
            / max(interaction_frames, 1),
            "multi_local_rate": stats["interaction_multi_local"]
            / max(interaction_frames, 1),
            "max_concurrent_interaction": stats["max_concurrent_interaction"],
            "collision_pre10_interaction_frames": stats[
                "collision_pre10_interaction"
            ],
            "collision_pre10_interaction_linear": summarize_values(
                stats["collision_pre10_linear"]
            ),
        }
    return by_scenario


def weighted_behavior(scenario_ids, trajectories):
    totals = defaultdict(float)
    lists = defaultdict(list)
    for scenario_id in scenario_ids:
        item = trajectories[scenario_id]
        frames = item["interaction_agent_frames"]
        totals["episodes"] += 1
        totals["active_agent_frames"] += item["active_agent_frames"]
        totals["interaction_agent_frames"] += frames
        totals["interaction_segments"] += item["interaction_segments"]
        totals["high_speed_close"] += item["high_speed_close_rate"] * frames
        totals["negative_progress"] += item["negative_progress_rate"] * frames
        totals["stagnant"] += item["stagnant_rate"] * frames
        totals["multi_local"] += item["multi_local_rate"] * frames
        for metric in (
            "interaction_linear",
            "interaction_abs_angular",
            "interaction_progress",
            "interaction_nearest",
            "interaction_local_degree",
        ):
            value = item[metric]["mean"]
            if value is not None:
                lists[metric].append((value, frames))
    frames = totals["interaction_agent_frames"]
    output = {
        "episodes": int(totals["episodes"]),
        "interaction_agent_frames": int(frames),
        "interaction_share": frames / max(totals["active_agent_frames"], 1),
        "segments_per_episode": totals["interaction_segments"]
        / max(totals["episodes"], 1),
        "high_speed_close_rate": totals["high_speed_close"] / max(frames, 1),
        "negative_progress_rate": totals["negative_progress"] / max(frames, 1),
        "stagnant_rate": totals["stagnant"] / max(frames, 1),
        "multi_local_rate": totals["multi_local"] / max(frames, 1),
    }
    for metric, values in lists.items():
        metric_name = metric[len("interaction_") :]
        output[f"mean_{metric_name}"] = sum(
            value * weight for value, weight in values
        ) / max(sum(weight for _, weight in values), 1)
    return output


def relation(new, old):
    if new["full_success"] > old["full_success"]:
        return "ie2_improved"
    if new["full_success"] < old["full_success"]:
        return "ie2_degraded"
    return "tied"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--old-result", required=True, type=Path)
    parser.add_argument("--new-result", required=True, type=Path)
    parser.add_argument("--old-trajectory", required=True, type=Path)
    parser.add_argument("--new-trajectory", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    metadata = load_manifest(args.manifest)
    expected_ids = list(metadata)
    old_results = load_results(args.old_result, expected_ids)
    new_results = load_results(args.new_result, expected_ids)
    old_trajectories = parse_trajectory(args.old_trajectory)
    new_trajectories = parse_trajectory(args.new_trajectory)
    for label, trajectories in (
        ("old", old_trajectories),
        ("new", new_trajectories),
    ):
        if set(trajectories) != set(expected_ids):
            raise ValueError(f"{label} trajectory scenario audit failed")

    groups = {
        "all": expected_ids,
        "zero": [item for item in expected_ids if metadata[item]["topology"] == "zero"],
        "edge1": [
            item for item in expected_ids if metadata[item]["topology"] == "edge1"
        ],
        "multi": [
            item for item in expected_ids if metadata[item]["topology"] == "multi"
        ],
    }
    relations = defaultdict(list)
    case_rows = []
    for scenario_id in expected_ids:
        outcome_relation = relation(new_results[scenario_id], old_results[scenario_id])
        relations[outcome_relation].append(scenario_id)
        row = {
            "scenario_id": scenario_id,
            **metadata[scenario_id],
            "relation": outcome_relation,
        }
        for prefix, result, trajectory in (
            ("old", old_results[scenario_id], old_trajectories[scenario_id]),
            ("ie2", new_results[scenario_id], new_trajectories[scenario_id]),
        ):
            row.update(
                {
                    f"{prefix}_full_success": result["full_success"],
                    f"{prefix}_collision": result["collision"],
                    f"{prefix}_unresolved": result["unresolved"],
                    f"{prefix}_timeout": result["timeout"],
                    f"{prefix}_steps": result["steps"],
                    f"{prefix}_interaction_share": trajectory["interaction_share"],
                    f"{prefix}_interaction_segments": trajectory[
                        "interaction_segments"
                    ],
                    f"{prefix}_mean_linear": trajectory["interaction_linear"]["mean"],
                    f"{prefix}_mean_abs_angular": trajectory[
                        "interaction_abs_angular"
                    ]["mean"],
                    f"{prefix}_mean_progress": trajectory["interaction_progress"][
                        "mean"
                    ],
                    f"{prefix}_mean_nearest": trajectory["interaction_nearest"][
                        "mean"
                    ],
                    f"{prefix}_high_speed_close_rate": trajectory[
                        "high_speed_close_rate"
                    ],
                    f"{prefix}_negative_progress_rate": trajectory[
                        "negative_progress_rate"
                    ],
                    f"{prefix}_stagnant_rate": trajectory["stagnant_rate"],
                    f"{prefix}_multi_local_rate": trajectory["multi_local_rate"],
                    f"{prefix}_max_concurrent_interaction": trajectory[
                        "max_concurrent_interaction"
                    ],
                }
            )
        case_rows.append(row)

    behavior = {
        group: {
            "old_epoch16": weighted_behavior(ids, old_trajectories),
            "ie2": weighted_behavior(ids, new_trajectories),
        }
        for group, ids in groups.items()
    }
    relation_behavior = {
        name: {
            "count": len(ids),
            "by_topology": {
                topology: sum(metadata[item]["topology"] == topology for item in ids)
                for topology in ("zero", "edge1", "multi")
            },
            "old_epoch16": weighted_behavior(ids, old_trajectories),
            "ie2": weighted_behavior(ids, new_trajectories),
        }
        for name, ids in relations.items()
    }
    payload = {
        "audit": {
            "manifest_scenarios": len(expected_ids),
            "old_trajectory_scenarios": len(old_trajectories),
            "new_trajectory_scenarios": len(new_trajectories),
            "status": "passed",
        },
        "behavior_by_topology": behavior,
        "behavior_by_outcome_relation": relation_behavior,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "diagnosis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "paired_cases.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_rows[0]))
        writer.writeheader()
        writer.writerows(case_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
