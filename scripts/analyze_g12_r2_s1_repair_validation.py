#!/usr/bin/env python3
import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def load_manifest_ids(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [str(item["scenario_id"]) for item in payload["scenarios"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = list(np.load(args.stats, allow_pickle=True))
    expected_ids = load_manifest_ids(args.manifest)
    observed_ids = [str(record[12]) for record in records]
    if len(records) != len(expected_ids):
        raise SystemExit(
            f"Expected {len(expected_ids)} episodes, found {len(records)}"
        )
    if observed_ids != expected_ids:
        mismatch = next(
            index
            for index, (observed, expected) in enumerate(
                zip(observed_ids, expected_ids), start=1
            )
            if observed != expected
        )
        raise SystemExit(
            "Manifest order mismatch at episode %i: observed=%s expected=%s"
            % (mismatch, observed_ids[mismatch - 1], expected_ids[mismatch - 1])
        )
    if len(set(observed_ids)) != len(observed_ids):
        raise SystemExit("Validation contains duplicate scenario IDs")

    episodes = len(records)
    successes = sum(int(record[6]) for record in records)
    collisions = sum(int(record[7]) for record in records)
    unresolved = sum(int(record[10]) for record in records)
    full_successes = sum(int(record[8]) for record in records)
    timeouts = sum(int(record[11]) for record in records)
    if successes + collisions + unresolved != episodes:
        raise SystemExit("Terminal outcome accounting is inconsistent")

    admission = {
        "full_success": full_successes >= 117,
        "collision": collisions <= 3,
        "timeout": timeouts <= 3,
    }
    summary = {
        "experiment": "G12-R2-S1-repair-broad-validation",
        "episodes": episodes,
        "successes": successes,
        "collisions": collisions,
        "unresolved": unresolved,
        "full_successes": full_successes,
        "timeouts": timeouts,
        "success_rate": successes / episodes,
        "collision_rate": collisions / episodes,
        "unresolved_rate": unresolved / episodes,
        "full_success_rate": full_successes / episodes,
        "timeout_episode_rate": timeouts / episodes,
        "average_episode_steps": float(np.mean([record[3] for record in records])),
        "average_final_distance": float(np.mean([record[9] for record in records])),
        "manifest_order_audit": "passed",
        "admission": admission,
        "admission_passed": all(admission.values()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
