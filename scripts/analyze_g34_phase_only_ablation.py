#!/usr/bin/env python3
"""Analyze the same-checkpoint phase-only ablation against full G34."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_g27_p4_test import bca_mean_interval, sign_flip_p


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/03_保留专门化/02_论文主线"
RUN = BASE / "37_最终双监督消融"
RESULTS = RUN / "local_data/matched/results"
FULL_RESULTS = BASE / "36_G34_G25slice_matched/local_data/matched/results"
MANIFEST = BASE / "25_最终消融与Sealed评测/local_data/sealed_manifest/dense_test_first256.json.gz"
OUTPUT = RUN / "local_data/matched/statistics.json"
SEEDS = (20260901, 20260902, 20260903)
SCENES = 256
AGENTS = 5


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(path, expected):
    rows = np.load(path, allow_pickle=True)
    if rows.shape != (SCENES, 17):
        raise ValueError("wrong result shape: %s" % path)
    if [str(item) for item in rows[:, 12]] != expected:
        raise ValueError("scenario order mismatch: %s" % path)
    terminal = sum(int(row[6]) + int(row[7]) + int(row[10]) for row in rows)
    if terminal != SCENES * AGENTS:
        raise ValueError("terminal accounting mismatch: %s" % path)
    return rows


def aggregate(rows):
    return {
        "episodes": int(len(rows)),
        "full_success": float(rows[:, 8].astype(float).mean()),
        "agent_success": float(rows[:, 6].astype(float).sum() / (len(rows) * AGENTS)),
        "robot_collision": float(rows[:, 7].astype(float).sum() / (len(rows) * AGENTS)),
        "robot_unresolved": float(rows[:, 10].astype(float).sum() / (len(rows) * AGENTS)),
        "episode_timeout": float(rows[:, 11].astype(float).mean()),
        "environment_steps": float(rows[:, 3].astype(float).mean()),
        "interaction_selection_share": float(rows[:, 14].astype(float).mean()),
        "switches_per_episode": float(rows[:, 15].astype(float).mean()),
    }


def effect(scene_values, include_sign_flip=False):
    result = {
        "mean_difference": float(scene_values.mean()),
        "scene_cluster_bca_95_ci": bca_mean_interval(scene_values, seed=20260913),
    }
    if include_sign_flip:
        result["exploratory_sign_flip_two_sided_p"] = sign_flip_p(
            scene_values, seed=20260913
        )
    return result


def main():
    with gzip.open(MANIFEST, "rt", encoding="utf-8") as handle:
        expected = [str(item["scenario_id"]) for item in json.load(handle)["scenarios"]]
    if len(expected) != SCENES:
        raise ValueError("manifest is not 256 scenes")

    runs = {}
    hashes = {}
    for seed in SEEDS:
        paths = {
            "full": FULL_RESULTS / ("g34_g25slice_s%d.npy" % seed),
            "phase_only": RESULTS / ("g34_phase_only_s%d.npy" % seed),
        }
        for method, path in paths.items():
            runs[(method, seed)] = validate(path, expected)
            hashes["%s_s%d" % (method, seed)] = sha256(path)

    overall = {
        method: aggregate(np.concatenate([runs[(method, seed)] for seed in SEEDS]))
        for method in ("full", "phase_only")
    }
    effects = {}
    for name, column, divisor in (
        ("full_success", 8, 1.0),
        ("agent_success", 6, AGENTS),
        ("robot_collision", 7, AGENTS),
        ("environment_steps", 3, 1.0),
        ("interaction_selection_share", 14, 1.0),
        ("switches_per_episode", 15, 1.0),
    ):
        paired = np.stack(
            [
                (runs[("full", seed)][:, column].astype(float)
                 - runs[("phase_only", seed)][:, column].astype(float)) / divisor
                for seed in SEEDS
            ],
            axis=1,
        ).mean(axis=1)
        effects[name] = effect(paired, include_sign_flip=(name == "full_success"))

    output = {
        "format_version": 1,
        "experiment_id": "G34-final-phase-only-ablation",
        "comparison": "full dual supervision minus same-checkpoint phase-only decision",
        "manifest_sha256": sha256(MANIFEST),
        "result_sha256": hashes,
        "overall": overall,
        "full_minus_phase_only": effects,
        "interpretation_boundary": (
            "Post-main-evaluation component ablation on the matched G25 slice; "
            "not an independent confirmatory test."
        ),
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
