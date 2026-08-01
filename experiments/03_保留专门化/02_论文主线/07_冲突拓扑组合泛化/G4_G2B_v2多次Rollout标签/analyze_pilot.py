#!/usr/bin/env python3
"""Summarize G2-B v2 multi-rollout label-stability shards."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


LABEL_AMBIGUOUS = -1
LABEL_GENERALIST = 0
LABEL_STRONG = 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-scenarios", type=int)
    parser.add_argument("--expected-rollouts", type=int)
    parser.add_argument("--expected-batches", type=int)
    return parser.parse_args()


def scalar(shard, key):
    return int(np.asarray(shard[key]).item())


def anchor_alignment_mask(repeatability, keys):
    columns = {str(key): index for index, key in enumerate(keys.tolist())}
    required = {
        "anchor_position_error",
        "anchor_yaw_error",
        "anchor_linear_velocity_error",
        "anchor_angular_velocity_error",
        "anchor_active_mask_match",
    }
    missing = required - set(columns)
    if missing:
        raise ValueError(f"repeatability data is missing {sorted(missing)}")
    return (
        (repeatability[:, columns["anchor_position_error"]] <= 0.02)
        & (repeatability[:, columns["anchor_yaw_error"]] <= 0.02)
        & (repeatability[:, columns["anchor_linear_velocity_error"]] <= 0.02)
        & (repeatability[:, columns["anchor_angular_velocity_error"]] <= 0.03)
        & (repeatability[:, columns["anchor_active_mask_match"]] == 1.0)
    )


def main():
    args = parse_args()
    paths = sorted(args.shard_dir.glob("*.npz"))
    if not paths:
        raise ValueError(f"No shards found in {args.shard_dir}")
    if args.expected_scenarios is not None and len(paths) != args.expected_scenarios:
        raise ValueError(
            f"Found {len(paths)} scenarios; expected {args.expected_scenarios}"
        )

    labels = []
    reasons = []
    batch_labels = []
    alignments = []
    scenario_ids = []
    protocols = set()
    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            if scalar(shard, "format_version") != 3:
                raise ValueError(f"{path} is not a format-v3 distribution shard")
            protocol = (
                scalar(shard, "rollouts_per_actor"),
                scalar(shard, "label_batches"),
                scalar(shard, "horizon"),
                scalar(shard, "bootstrap_resamples"),
            )
            protocols.add(protocol)
            labels.append(shard["labels"].astype(int))
            reasons.extend(str(item) for item in shard["label_reasons"])
            batch_labels.append(shard["distribution_batch_labels"].astype(int))
            alignments.append(
                anchor_alignment_mask(
                    shard["repeatability"], shard["repeatability_keys"]
                )
            )
            scenario_ids.append(str(shard["scenario_id"].item()))
    if len(protocols) != 1:
        raise ValueError(f"Mixed collection protocols: {sorted(protocols)}")
    rollouts, batches, horizon, bootstrap_resamples = next(iter(protocols))
    if args.expected_rollouts is not None and rollouts != args.expected_rollouts:
        raise ValueError(f"Found {rollouts} rollouts per actor; expected {args.expected_rollouts}")
    if args.expected_batches is not None and batches != args.expected_batches:
        raise ValueError(f"Found {batches} batches; expected {args.expected_batches}")

    labels = np.concatenate(labels)
    batch_labels = np.concatenate(batch_labels)
    alignments = np.concatenate(alignments)
    stable_non_ambiguous = labels != LABEL_AMBIGUOUS
    any_batch_non_ambiguous = np.any(batch_labels != LABEL_AMBIGUOUS, axis=1)
    all_batches_agree = np.all(batch_labels == batch_labels[:, :1], axis=1)
    repeated_non_ambiguous = all_batches_agree & (
        batch_labels[:, 0] != LABEL_AMBIGUOUS
    )
    conditional_denominator = int(any_batch_non_ambiguous.sum())
    class_counts = Counter(int(label) for label in labels[stable_non_ambiguous])
    anchors = len(labels)
    checks = {
        "anchor_alignment_at_least_90_percent": float(alignments.mean()) >= 0.90,
        "stable_non_ambiguous_at_least_25_percent": (
            float(stable_non_ambiguous.mean()) >= 0.25
        ),
        "conditional_batch_agreement_at_least_70_percent": bool(
            conditional_denominator > 0
            and float(repeated_non_ambiguous.sum() / conditional_denominator) >= 0.70
        ),
        "both_actor_classes_have_at_least_two_anchors": bool(
            class_counts[LABEL_GENERALIST] >= 2 and class_counts[LABEL_STRONG] >= 2
        ),
    }
    result = {
        "result_version": 1,
        "status": "complete",
        "protocol": {
            "scenarios": len(paths),
            "anchors": anchors,
            "rollouts_per_actor": rollouts,
            "label_batches": batches,
            "horizon": horizon,
            "bootstrap_resamples": bootstrap_resamples,
            "scenario_ids": scenario_ids,
        },
        "anchor_alignment": {
            "count": int(alignments.sum()),
            "rate": float(alignments.mean()),
        },
        "stable_non_ambiguous": {
            "count": int(stable_non_ambiguous.sum()),
            "rate": float(stable_non_ambiguous.mean()),
            "generalist": class_counts[LABEL_GENERALIST],
            "strong": class_counts[LABEL_STRONG],
        },
        "batch_stability": {
            "anchors_with_any_non_ambiguous_batch": conditional_denominator,
            "repeated_non_ambiguous": int(repeated_non_ambiguous.sum()),
            "conditional_agreement": (
                float(repeated_non_ambiguous.sum() / conditional_denominator)
                if conditional_denominator
                else None
            ),
        },
        "final_reason_counts": dict(sorted(Counter(reasons).items())),
        "admission_checks": checks,
        "admission_pass": bool(all(checks.values())),
    }
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
