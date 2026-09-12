#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from g27_counterfactual import canonical_sha256
from g27_dataset import audit_split, records_to_arrays, sha256_file


G27 = (
    ROOT
    / "experiments/03_保留专门化/02_论文主线/27_反事实Reward增强Gate监督"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Audit completed G27 P1 data.")
    parser.add_argument(
        "--train-selection",
        type=Path,
        default=G27 / "local_data/protocol/p1_train_anchors.json",
    )
    parser.add_argument(
        "--validation-selection",
        type=Path,
        default=G27 / "local_data/protocol/p1_validation_anchors.json",
    )
    parser.add_argument(
        "--train-collection",
        type=Path,
        default=G27 / "local_data/counterfactual/train",
    )
    parser.add_argument(
        "--validation-collection",
        type=Path,
        default=G27 / "local_data/counterfactual/validation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=G27 / "local_data/counterfactual/audit",
    )
    return parser.parse_args()


def atomic_save_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def main():
    args = parse_args()
    train_records, train_summary = audit_split(
        args.train_selection, args.train_collection
    )
    validation_records, validation_summary = audit_split(
        args.validation_selection, args.validation_collection
    )
    train_scenes = {item["scenario_id"] for item in train_records}
    validation_scenes = {item["scenario_id"] for item in validation_records}
    overlap = sorted(train_scenes & validation_scenes)
    if overlap:
        raise ValueError("usable train/validation scenes overlap: %s" % overlap[:5])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.npz"
    validation_path = args.output_dir / "validation.npz"
    atomic_save_npz(train_path, records_to_arrays(train_records))
    atomic_save_npz(validation_path, records_to_arrays(validation_records))

    train_minimum = 192
    validation_minimum = 40
    train_cell_minimum = 8
    validation_cell_minimum = 3
    criteria = {
        "train_at_least_192": len(train_records) >= train_minimum,
        "validation_at_least_40": len(validation_records) >= validation_minimum,
        "train_each_cell_at_least_8": min(
            train_summary["coverage"]["usable"].values(), default=0
        )
        >= train_cell_minimum,
        "validation_each_cell_at_least_3": min(
            validation_summary["coverage"]["usable"].values(), default=0
        )
        >= validation_cell_minimum,
        "train_validation_scenes_disjoint": True,
    }
    summary = {
        "format_version": 1,
        "protocol": "G27-P1-audit-v1",
        "train": train_summary,
        "validation": validation_summary,
        "datasets": {
            "train": {"path": str(train_path), "sha256": sha256_file(train_path)},
            "validation": {
                "path": str(validation_path),
                "sha256": sha256_file(validation_path),
            },
        },
        "coverage_criteria": criteria,
        "coverage_passed": all(criteria.values()),
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    summary_path = args.output_dir / "summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["coverage_passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
