#!/usr/bin/env python3
"""Recover complete G17 result rows from an uninterrupted episode log."""

import argparse
import gzip
import json
import re
from pathlib import Path

import numpy as np


ROW = re.compile(
    r"Episode (?P<episode>\d+) complete \| case=(?P<case>[^ ]+) \| "
    r"env_steps=(?P<env_steps>\d+) \| agent_samples=(?P<agent_samples>\d+) \| "
    r"episode_env_steps=(?P<episode_env_steps>\d+) \| "
    r"episode_agent_samples=(?P<episode_agent_samples>\d+) \| "
    r"mean_reward=(?P<mean_reward>[-+0-9.eE]+) \| "
    r"success=(?P<success>\d+)/\d+ \| collision=(?P<collision>\d+)/\d+ \| "
    r"unresolved=(?P<unresolved>\d+)/\d+ \| full_success=(?P<full_success>\d+) \| "
    r"timeout=(?P<timeout>\d+) \| "
    r"mean_final_distance=(?P<mean_final_distance>[-+0-9.eE]+) \| "
    r"dense_action_share=(?P<dense_action_share>[-+0-9.eE]+) \| "
    r"gate_switches=(?P<gate_switches>\d+) \| "
    r"gate_mean_probability=(?P<gate_mean_probability>[-+0-9.eE]+)"
)


def load_ids(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [str(item["scenario_id"]) for item in json.load(handle)["scenarios"]]


def recover(log_path: Path, manifest_path: Path, output_path: Path):
    expected = load_ids(manifest_path)
    rows = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ROW.search(line)
        if not match:
            continue
        item = match.groupdict()
        rows.append([
            int(item["episode"]),
            int(item["env_steps"]),
            int(item["agent_samples"]),
            int(item["episode_env_steps"]),
            int(item["episode_agent_samples"]),
            float(item["mean_reward"]),
            int(item["success"]),
            int(item["collision"]),
            int(item["full_success"]),
            float(item["mean_final_distance"]),
            int(item["unresolved"]),
            int(item["timeout"]),
            item["case"],
            0,
            float(item["dense_action_share"]),
            int(item["gate_switches"]),
            float(item["gate_mean_probability"]),
        ])

    if len(rows) != len(expected):
        raise SystemExit(f"expected {len(expected)} complete rows, found {len(rows)}")
    if [str(row[12]) for row in rows] != expected:
        raise SystemExit("recovered scenario IDs do not match manifest order")
    if [row[0] for row in rows] != list(range(1, len(expected) + 1)):
        raise SystemExit("episode numbers are incomplete or duplicated")
    if sum(row[6] + row[7] + row[10] for row in rows) != len(expected) * 5:
        raise SystemExit("terminal accounting mismatch")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.asarray(rows, dtype=object))
    print(f"recovered {len(rows)} rows -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recover(args.log, args.manifest, args.output)


if __name__ == "__main__":
    main()
