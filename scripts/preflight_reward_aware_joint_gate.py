#!/usr/bin/env python3
"""Low-cost preflight audit for the advisor's joint y/Delta-r Gate idea."""

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED = {
    "gate_windows",
    "history_lengths",
    "interaction_labels",
    "delta_r_1",
    "delta_r_2",
    "mean_delta_r",
}


def audit_split(path: Path, expected_gate_frames: int, reliable_threshold: float):
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(REQUIRED - set(data.files))
        if missing:
            raise ValueError(f"{path} is missing {missing}")
        labels = np.asarray(data["interaction_labels"], dtype=np.float64)
        delta = np.asarray(data["mean_delta_r"], dtype=np.float64)
        delta_1 = np.asarray(data["delta_r_1"], dtype=np.float64)
        delta_2 = np.asarray(data["delta_r_2"], dtype=np.float64)
        windows = np.asarray(data["gate_windows"])
        history = np.asarray(data["history_lengths"])
        strata = np.asarray(data["strata"])

    if windows.ndim != 3 or windows.shape[1:] != (8, 82):
        raise ValueError(f"unexpected gate_windows shape: {windows.shape}")
    if len(labels) != len(delta) or len(delta) != len(windows):
        raise ValueError("sample arrays have inconsistent lengths")
    if not np.isfinite(delta).all() or not np.isfinite(delta_1).all() or not np.isfinite(delta_2).all():
        raise ValueError("reward arrays contain non-finite values")
    if not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("interaction labels are not binary")
    reliable = np.abs(delta) > reliable_threshold
    agreement = ((delta[reliable] > 0.0) == (labels[reliable] > 0.5)) if reliable.any() else np.array([], dtype=bool)

    by_stratum = {}
    for stratum in sorted(set(strata.tolist())):
        mask = strata == stratum
        by_stratum[stratum] = {
            "samples": int(mask.sum()),
            "interaction_label_rate": float(labels[mask].mean()),
            "positive_delta_rate": float((delta[mask] > 0.0).mean()),
            "reliable_delta_count": int((reliable & mask).sum()),
        }

    return {
        "path": str(path),
        "samples_with_reward": int(len(delta)),
        "expected_gate_frames": int(expected_gate_frames),
        "reward_coverage": float(len(delta) / expected_gate_frames) if expected_gate_frames else None,
        "label_counts": {
            "standard": int((labels == 0.0).sum()),
            "interaction": int((labels == 1.0).sum()),
        },
        "delta_quantiles": {
            str(q): float(np.quantile(delta, q)) for q in [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]
        },
        "delta_sign_counts": {
            "positive": int((delta > 0.0).sum()),
            "negative": int((delta < 0.0).sum()),
            "zero": int((delta == 0.0).sum()),
            "abs_gt_1": int((np.abs(delta) > 1.0).sum()),
            "abs_gt_10": int((np.abs(delta) > 10.0).sum()),
            "reliable": int(reliable.sum()),
        },
        "branch_consistency": {
            "max_abs_delta_r1_minus_delta_r2": float(np.max(np.abs(delta_1 - delta_2))),
            "mean_abs_delta_r1_minus_delta_r2": float(np.mean(np.abs(delta_1 - delta_2))),
        },
        "label_reward_direction": {
            "reliable_threshold": float(reliable_threshold),
            "agreement_count": int(agreement.sum()),
            "reliable_count": int(len(agreement)),
            "agreement_rate": float(agreement.mean()) if len(agreement) else None,
        },
        "history_length": {
            "minimum": int(history.min()),
            "maximum": int(history.max()),
        },
        "by_stratum": by_stratum,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--train-gate-frames", type=int, required=True)
    parser.add_argument("--validation-gate-frames", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reliable-threshold", type=float, default=0.005)
    args = parser.parse_args()
    if args.reliable_threshold < 0.0:
        raise ValueError("reliable threshold must be non-negative")
    report = {
        "protocol": "joint-y-delta-r-preflight-v1",
        "status": "preflight_only",
        "reliable_threshold": args.reliable_threshold,
        "train": audit_split(args.train, args.train_gate_frames, args.reliable_threshold),
        "validation": audit_split(args.validation, args.validation_gate_frames, args.reliable_threshold),
    }
    report["stop_lines"] = {
        "full_gate_sample_reward_coverage_required": True,
        "current_coverage_sufficient_for_full_joint_training": False,
        "reward_direction_agreement_sufficient_for_label_substitution": False,
        "long_run_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
