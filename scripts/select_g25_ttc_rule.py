#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))
sys.path.insert(0, str(ROOT / "scripts"))

from robot_perception.dataset import list_shards
from robot_perception.models import LocalRobotDetector
from train_g11_b_aggregated_gate import G11_B_STUDENT_DATASET_SHA256
from train_temporal_interaction_gate import (
    EXPECTED_A1_MANIFEST_SHA256,
    EXPECTED_DETECTOR_SHA256,
    EXPECTED_INTERACTION_ACTOR_SHA256,
    EXPECTED_STANDARD_ACTOR_SHA256,
    build_dataset,
    load_actor,
    sha256_file,
    verify_frozen_actor,
)


BASE = ROOT / "experiments/03_保留专门化/02_论文主线"
GATE_ROOT = BASE / "11_可部署在线Gate研究"
A1 = GATE_ROOT / "G11_A1_当前协议时序pilot"
B2 = GATE_ROOT / "G11_B_student_rollout_v1"
VIEW = BASE / "datasets/fixed_v1/views/g11_a1_gate_v1"


def parse_args():
    parser = argparse.ArgumentParser(description="Freeze the G25 TTC/CPA rule on train data.")
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE / "25_最终消融与Sealed评测/local_data/ttc_rule/selection.json",
    )
    parser.add_argument("--zero-edge-fpr-cap", type=float, default=0.10)
    return parser.parse_args()


def load_edges(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        scenarios = json.load(handle)["scenarios"]
    return {
        str(item["scenario_id"]): int(item["metrics"]["conflict_edge_count"])
        for item in scenarios
    }


def candidate_rule(base_features, config):
    tracks = base_features[:, 24 : 24 + 4 * 11].reshape(-1, 4, 11)
    present = tracks[:, :, 0] > 0.5
    shape = np.maximum(tracks[:, :, 1], tracks[:, :, 2])
    ranges = tracks[:, :, 3] * 4.0
    closing = tracks[:, :, 7] * 1.5
    ttc = tracks[:, :, 8] * 4.0
    cpa = tracks[:, :, 9] * 4.0
    age = tracks[:, :, 10] * 5.0
    candidates = (
        present
        & (shape >= config["shape_probability"])
        & (age >= config["minimum_age"] - 1e-6)
        & (ranges <= config["maximum_range"])
        & (closing >= config["minimum_closing_speed"])
        & (ttc <= config["maximum_ttc"])
        & (cpa <= config["maximum_cpa_distance"])
    )
    return np.any(candidates, axis=1)


def main():
    args = parse_args()
    if not 0.0 < args.zero_edge_fpr_cap < 1.0:
        raise ValueError("zero-edge FPR cap must lie within (0, 1)")
    manifests = {
        "train": sha256_file(VIEW / "train.json.gz"),
        "validation": sha256_file(VIEW / "validation.json.gz"),
    }
    if manifests != EXPECTED_A1_MANIFEST_SHA256:
        raise ValueError("A1 manifest hash mismatch")
    audit = json.loads((B2 / "local_data/train_audit.json").read_text())
    if audit.get("dataset_sha256") != G11_B_STUDENT_DATASET_SHA256:
        raise ValueError("student dataset hash mismatch")

    detector_path = BASE / "results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"
    standard_path = ROOT / "TD3/pytorch_models/TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best_actor.pth"
    interaction_path = ROOT / "TD3/pytorch_models/interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016_actor.pth"
    frozen_hashes = {
        "detector": verify_frozen_actor(detector_path, EXPECTED_DETECTOR_SHA256),
        "standard_actor": verify_frozen_actor(standard_path, EXPECTED_STANDARD_ACTOR_SHA256),
        "interaction_actor": verify_frozen_actor(interaction_path, EXPECTED_INTERACTION_ACTOR_SHA256),
    }
    detector_payload = torch.load(detector_path, map_location="cpu", weights_only=False)
    detector = LocalRobotDetector(**detector_payload.get("model_config", {}))
    detector.load_state_dict(detector_payload["model_state_dict"])
    detector.eval()
    standard_actor = load_actor(standard_path, "cpu")
    interaction_actor = load_actor(interaction_path, "cpu")
    build_args = argparse.Namespace(
        max_tracks=4, max_candidates=12, batch_size=256, device="cpu"
    )

    sources = []
    for name, directory in (
        ("a1_5a", A1 / "local_data/shards/train"),
        ("student", B2 / "local_data/student_shards/train"),
    ):
        dataset = build_dataset(
            list_shards(directory), detector, standard_actor, interaction_actor,
            build_args, "train"
        )
        sources.append((name, dataset))
    features = np.concatenate([item.base_features for _, item in sources])
    labels = np.concatenate([item.labels["any"] for _, item in sources]).astype(bool)
    scenarios = np.concatenate([item.scenarios for _, item in sources])
    edge_counts = load_edges(VIEW / "train.json.gz")
    zero_negative = np.asarray(
        [edge_counts[str(item)] == 0 for item in scenarios], dtype=bool
    ) & ~labels
    if not np.any(zero_negative) or not np.any(labels):
        raise ValueError("train data lacks zero-edge negatives or positive labels")

    grids = {
        "shape_probability": (0.3, 0.4, 0.5, 0.6, 0.7),
        "minimum_age": (2, 3),
        "maximum_range": (2.0, 2.5, 3.0),
        "minimum_closing_speed": (0.05, 0.1, 0.2),
        "maximum_ttc": (1.5, 2.0, 3.0, 3.5),
        "maximum_cpa_distance": (0.75, 1.0, 1.25),
    }
    candidates = []
    keys = tuple(grids)
    for values in itertools.product(*(grids[key] for key in keys)):
        config = dict(zip(keys, values))
        prediction = candidate_rule(features, config)
        zero_fpr = float(np.mean(prediction[zero_negative]))
        if zero_fpr > args.zero_edge_fpr_cap + 1e-12:
            continue
        recall = float(np.mean(prediction[labels]))
        overall_fpr = float(np.mean(prediction[~labels]))
        candidates.append((recall, -overall_fpr, -zero_fpr, config, prediction))
    if not candidates:
        raise ValueError("no TTC/CPA rule satisfies the frozen zero-edge FPR cap")
    recall, neg_fpr, neg_zero_fpr, selected, prediction = max(
        candidates, key=lambda item: (item[0], item[1], item[2])
    )
    stay = {
        "shape_probability": round(
            max(selected["shape_probability"] - 0.1, 0.0), 6
        ),
        "minimum_age": selected["minimum_age"],
        "maximum_range": min(selected["maximum_range"] + 0.5, 4.0),
        "minimum_closing_speed": selected["minimum_closing_speed"] / 2.0,
        "maximum_ttc": min(selected["maximum_ttc"] + 0.5, 4.0),
        "maximum_cpa_distance": min(selected["maximum_cpa_distance"] + 0.25, 1.5),
    }
    result = {
        "protocol": {
            "experiment_id": "G25-V3-TTC-train-selection",
            "sealed_test_read": False,
            "dense_validation_read": False,
            "selection_order": "zero-edge negative FPR cap, then maximum interaction-label recall, then minimum overall FPR",
            "zero_edge_fpr_cap": args.zero_edge_fpr_cap,
            "grid": grids,
            "minimum_hold_steps": 3,
            "evaluation_stride": 2,
            "frozen_hashes": frozen_hashes,
            "manifest_hashes": manifests,
            "student_dataset_sha256": G11_B_STUDENT_DATASET_SHA256,
        },
        "samples": {
            "frames": int(len(features)),
            "positive_frames": int(np.sum(labels)),
            "zero_edge_negative_frames": int(np.sum(zero_negative)),
            "sources": {
                name: {"frames": len(item.base_features), "scenarios": len(set(item.scenarios.tolist()))}
                for name, item in sources
            },
        },
        "selected": {
            "enter": selected,
            "stay": stay,
            "train_recall": recall,
            "train_overall_fpr": -neg_fpr,
            "train_zero_edge_fpr": -neg_zero_fpr,
            "predicted_positive_rate": float(np.mean(prediction)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({"output": str(args.output), "sha256": digest, **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
