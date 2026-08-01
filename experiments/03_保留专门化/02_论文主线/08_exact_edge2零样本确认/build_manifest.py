#!/usr/bin/env python3
"""Build a fixed exact-edge-2 confirmation set outside G3 development scenes."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def parse_args():
    main_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=main_dir / "datasets/fixed_v1/dense/validation.json.gz",
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        default=main_dir
        / "datasets/fixed_v1/views/dense_validation_monitor_v1/validation.json.gz",
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("validation.json")
    )
    parser.add_argument("--scenarios", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


def load_json(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def rank(case, seed):
    value = f"{seed}:{case['scenario_id']}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def main():
    args = parse_args()
    if args.scenarios < 1:
        raise ValueError("scenarios must be positive")
    source = load_json(args.source)
    excluded = {
        str(case["scenario_id"])
        for case in load_json(args.exclude)["scenarios"]
    }
    candidates = [
        case
        for case in source["scenarios"]
        if str(case["scenario_id"]) not in excluded
        and int(case.get("metrics", {}).get("conflict_edge_count", 0)) == 2
    ]
    if len(candidates) < args.scenarios:
        raise ValueError(
            f"Only {len(candidates)} non-development exact-edge-2 cases are available"
        )
    candidates.sort(key=lambda case: rank(case, args.seed))
    selected = candidates[: args.scenarios]
    output = dict(source)
    output.update(
        {
            "dataset_id": "exact_edge2_zero_shot_confirmation_v1",
            "split": "validation_confirmation",
            "view_config": {
                "conflict_edge_count": 2,
                "exclude_view": "dense_validation_monitor_v1",
                "candidate_count": len(candidates),
                "selection": "sha256_rank",
                "selection_seed": args.seed,
                "scenarios": args.scenarios,
                "sealed_test_read": False,
            },
            "scenarios": selected,
        }
    )
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "source_scenarios": len(source["scenarios"]),
                "excluded_development_scenarios": len(excluded),
                "eligible_exact_edge2_scenarios": len(candidates),
                "selected_scenarios": len(selected),
                "selection_seed": args.seed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
