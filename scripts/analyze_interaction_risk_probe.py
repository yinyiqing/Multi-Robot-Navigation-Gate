#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


EPISODE_PATTERN = re.compile(
    r"Episode (?P<episode>\d+) complete \| case=(?P<case>[^ |]+).*?"
    r"success=(?P<success>\d+)/\d+ \| collision=(?P<collision>\d+)/\d+.*?"
    r"full_success=(?P<full_success>\d)"
)
RISK_BANDS = (
    ("deep", 0.0, 0.4),
    ("close", 0.4, 0.6),
    ("margin", 0.6, 0.9),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze a trajectory-recorded interaction-risk probe."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encounter-distance", type=float, default=1.2)
    parser.add_argument("--step-seconds", type=float, default=0.2)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def risk_band(scenario):
    separation = float(scenario["metrics"]["min_synchronized_path_separation_m"])
    for name, lower, upper in RISK_BANDS:
        if lower <= separation < upper:
            return name
    raise ValueError(f"Scenario separation is outside risk bands: {separation}")


def parse_outcomes(path):
    outcomes = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = EPISODE_PATTERN.search(line)
            if not match:
                continue
            values = match.groupdict()
            episode = int(values.pop("episode"))
            outcomes[episode] = {
                "case": values["case"],
                "success": int(values["success"]),
                "collision": int(values["collision"]),
                "full_success": int(values["full_success"]),
            }
    return outcomes


def parse_frames(path):
    frames = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            frames[int(record["episode"])].append(record)
    return frames


def mean_or_none(values):
    return statistics.mean(values) if values else None


def median_or_none(values):
    return statistics.median(values) if values else None


def summarize_group(rows):
    episodes = len(rows)
    agents = 5 * episodes
    return {
        "episodes": episodes,
        "agent_success": sum(row["success"] for row in rows) / agents,
        "collision": sum(row["collision"] for row in rows) / agents,
        "full_success": sum(row["full_success"] for row in rows) / episodes,
        "actual_pair_min_distance_mean_m": statistics.mean(
            row["actual_pair_min_distance_m"] for row in rows
        ),
        "actual_pair_min_distance_median_m": statistics.median(
            row["actual_pair_min_distance_m"] for row in rows
        ),
    }


def summarize_encounters(rows, step_seconds):
    selected = [row["encounter"] for row in rows if row["encounter"] is not None]
    fields = ("distance_m", "closing_speed_mps", "ttc_s", "mean_linear", "mean_abs_angular", "mean_lidar_min_m")
    summary = {"encounters": len(selected)}
    for field in fields:
        values = [item[field] for item in selected]
        summary[f"{field}_mean"] = mean_or_none(values)
        summary[f"{field}_median"] = median_or_none(values)
    summary["step_seconds"] = step_seconds
    return summary


def build_rows(scenarios, outcomes, frames, encounter_distance, step_seconds):
    rows = []
    for episode, outcome in sorted(outcomes.items()):
        scenario = scenarios[outcome["case"]]
        conflict_pair = scenario["metrics"]["conflict_edges"][0]["agents"]
        pair_indices = [list(scenario["agents"]).index(name) for name in conflict_pair]
        previous_distance = None
        pair_distances = []
        collision_names = set()
        encounter = None

        for frame in frames[episode]:
            positions = frame["positions"]
            distance = math.dist(
                positions[conflict_pair[0]], positions[conflict_pair[1]]
            )
            pair_distances.append(distance)
            collision_names.update(
                name
                for name, values in frame["agents"].items()
                if values["collision"]
            )
            if previous_distance is not None:
                closing_speed = (previous_distance - distance) / step_seconds
                if (
                    encounter is None
                    and distance <= encounter_distance
                    and closing_speed > 0.0
                ):
                    encounter = {
                        "distance_m": distance,
                        "closing_speed_mps": closing_speed,
                        "ttc_s": distance / closing_speed,
                        "mean_linear": statistics.mean(
                            frame["actions"][index][0] for index in pair_indices
                        ),
                        "mean_abs_angular": statistics.mean(
                            abs(frame["actions"][index][1]) for index in pair_indices
                        ),
                        "mean_lidar_min_m": statistics.mean(
                            min(frame["actor_states"][index][:20])
                            for index in pair_indices
                        ),
                    }
            previous_distance = distance

        pair_collision_members = len(set(conflict_pair) & collision_names)
        rows.append(
            {
                **outcome,
                "risk_band": risk_band(scenario),
                "pool": scenario["preset"],
                "offline_separation_m": scenario["metrics"][
                    "min_synchronized_path_separation_m"
                ],
                "conflict_pair": conflict_pair,
                "actual_pair_min_distance_m": min(pair_distances),
                "collision_names": sorted(collision_names),
                "conflict_pair_collision_members": pair_collision_members,
                "encounter": encounter,
            }
        )
    return rows


def main():
    args = parse_args()
    if args.encounter_distance <= 0.0:
        raise ValueError("--encounter-distance must be positive")
    if args.step_seconds <= 0.0:
        raise ValueError("--step-seconds must be positive")

    payload = load_json(args.manifest)
    scenarios = {item["scenario_id"]: item for item in payload["scenarios"]}
    outcomes = parse_outcomes(args.log)
    frames = parse_frames(args.trajectory)
    if set(outcomes) != set(frames):
        raise ValueError("Outcome episodes and trajectory episodes do not match")
    if {item["case"] for item in outcomes.values()} != set(scenarios):
        raise ValueError("Probe outcomes do not exactly cover the manifest")

    rows = build_rows(
        scenarios,
        outcomes,
        frames,
        args.encounter_distance,
        args.step_seconds,
    )
    failures = [row for row in rows if not row["full_success"]]
    result = {
        "protocol": {
            "manifest": str(args.manifest),
            "manifest_sha256": sha256(args.manifest),
            "log": str(args.log),
            "log_sha256": sha256(args.log),
            "trajectory": str(args.trajectory),
            "trajectory_sha256": sha256(args.trajectory),
            "encounter_distance_m": args.encounter_distance,
            "step_seconds": args.step_seconds,
        },
        "overall": summarize_group(rows),
        "by_risk_band": {
            band: summarize_group([row for row in rows if row["risk_band"] == band])
            for band, _, _ in RISK_BANDS
        },
        "by_pool": {
            pool: summarize_group([row for row in rows if row["pool"] == pool])
            for pool in ("standard", "dense")
        },
        "failure_attribution": {
            "failed_episodes": len(failures),
            "failures_with_any_conflict_pair_member_colliding": sum(
                row["conflict_pair_collision_members"] >= 1 for row in failures
            ),
            "failures_with_both_conflict_pair_members_colliding": sum(
                row["conflict_pair_collision_members"] == 2 for row in failures
            ),
        },
        "encounters": {
            "success": summarize_encounters(
                [row for row in rows if row["full_success"]], args.step_seconds
            ),
            "failure": summarize_encounters(failures, args.step_seconds),
        },
        "episodes": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in ("overall", "by_risk_band", "by_pool", "failure_attribution", "encounters")}, indent=2))


if __name__ == "__main__":
    main()
