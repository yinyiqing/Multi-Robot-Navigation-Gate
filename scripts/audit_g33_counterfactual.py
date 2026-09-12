#!/usr/bin/env python3
"""Audit a G33 counterfactual collection without applying G27's old size gate."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from g27_counterfactual import canonical_sha256
from g27_dataset import audit_split, records_to_arrays, sha256_file


def save_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-selection", type=Path, required=True)
    parser.add_argument("--validation-selection", type=Path, required=True)
    parser.add_argument("--train-collection", type=Path, required=True)
    parser.add_argument("--validation-collection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    train_records, train_summary = audit_split(args.train_selection, args.train_collection)
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
    save_npz(train_path, records_to_arrays(train_records))
    save_npz(validation_path, records_to_arrays(validation_records))

    train_cells = Counter(item["stratum"] for item in train_records)
    validation_cells = Counter(item["stratum"] for item in validation_records)
    criteria = {
        "train_has_records": bool(train_records),
        "validation_has_records": bool(validation_records),
        "train_each_cell_at_least_4": min(train_cells.values(), default=0) >= 4,
        "validation_each_cell_at_least_2": min(validation_cells.values(), default=0) >= 2,
        "train_validation_scenes_disjoint": not overlap,
    }
    summary = {
        "format_version": 1,
        "protocol": "G33-counterfactual-quality-audit-v1",
        "train": train_summary,
        "validation": validation_summary,
        "datasets": {
            "train": {"path": str(train_path), "sha256": sha256_file(train_path)},
            "validation": {"path": str(validation_path), "sha256": sha256_file(validation_path)},
        },
        "quality_criteria": criteria,
        "quality_passed": all(criteria.values()),
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["quality_passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
