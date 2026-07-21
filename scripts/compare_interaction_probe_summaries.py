#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare paired interaction-probe episode summaries."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--select-candidate-band", default="deep")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_episodes(path):
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {item["case"]: item for item in payload["episodes"]}


def paired_counts(baseline, candidate, case_ids):
    return {
        "both_success": sum(
            baseline[case]["full_success"] and candidate[case]["full_success"]
            for case in case_ids
        ),
        "baseline_only": sum(
            baseline[case]["full_success"] and not candidate[case]["full_success"]
            for case in case_ids
        ),
        "candidate_only": sum(
            not baseline[case]["full_success"] and candidate[case]["full_success"]
            for case in case_ids
        ),
        "neither_success": sum(
            not baseline[case]["full_success"] and not candidate[case]["full_success"]
            for case in case_ids
        ),
    }


def aggregate(rows):
    episodes = len(rows)
    return {
        "episodes": episodes,
        "agent_success": sum(item["success"] for item in rows) / (5 * episodes),
        "collision": sum(item["collision"] for item in rows) / (5 * episodes),
        "full_success": sum(item["full_success"] for item in rows) / episodes,
    }


def main():
    args = parse_args()
    baseline = load_episodes(args.baseline)
    candidate = load_episodes(args.candidate)
    if set(baseline) != set(candidate):
        raise ValueError("Baseline and candidate scenario IDs do not match")

    result = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "paired": {},
    }
    bands = sorted({item["risk_band"] for item in baseline.values()})
    for band in bands + ["all"]:
        case_ids = [
            case
            for case, item in baseline.items()
            if band == "all" or item["risk_band"] == band
        ]
        result["paired"][band] = paired_counts(baseline, candidate, case_ids)

    selected_rows = [
        (
            candidate[case]
            if item["risk_band"] == args.select_candidate_band
            else item
        )
        for case, item in baseline.items()
    ]
    result["selective_candidate"] = {
        "candidate_band": args.select_candidate_band,
        **aggregate(selected_rows),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
