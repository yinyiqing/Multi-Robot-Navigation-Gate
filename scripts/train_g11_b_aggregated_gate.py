#!/usr/bin/env python3
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from train_temporal_interaction_gate import (
    EXPECTED_A1_MANIFEST_SHA256,
    EXPECTED_DETECTOR_SHA256,
    EXPECTED_INTERACTION_ACTOR_SHA256,
    EXPECTED_STANDARD_ACTOR_SHA256,
    GateDataset,
    TemporalInteractionGate,
    build_dataset,
    evaluate_probabilities,
    load_actor,
    make_temporal_windows,
    predict,
    reset_random_seed,
    sha256_file,
    train_candidate,
    verify_frozen_actor,
)
from robot_perception.dataset import list_shards
from robot_perception.gate_features import gate_feature_dim
from robot_perception.models import LocalRobotDetector
from temporal_interaction_gate import ACTOR_COMPARISON_DIM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GATE_ROUTE = (
    PROJECT_ROOT
    / "experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究"
)
A1_ROUTE = GATE_ROUTE / "G11_A1_当前协议时序pilot"
B_ROUTE = GATE_ROUTE / "G11_B_student_rollout_v1"
VIEW_DIR = (
    PROJECT_ROOT
    / "experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views"
    / "g11_a1_gate_v1"
)
A1_MAIN_DIR = A1_ROUTE / "local_data/training/seed20260804"
A1_MAIN_SUMMARY_SHA256 = (
    "c0490131ae34826e8f80b8d503a874257556a8934d887c37569564f9b671768f"
)
A1_MAIN_T1_SHA256 = (
    "d9b05d9f86e5bad4d2071c041187b618ebca6f1a3cc1f9c46e8b14b1a451537a"
)
G11_B_STUDENT_DATASET_SHA256 = (
    "bda1a3ebe16eb481da8629b21f8f030fe9f0a6499da6409c90b0c2e936614fba"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the source-balanced G11-B2 aggregated temporal Gate."
    )
    parser.add_argument(
        "--a1-train-dir", type=Path, default=A1_ROUTE / "local_data/shards/train"
    )
    parser.add_argument(
        "--student-train-dir",
        type=Path,
        default=B_ROUTE / "local_data/student_shards/train",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=A1_ROUTE / "local_data/shards/validation",
    )
    parser.add_argument("--train-manifest", type=Path, default=VIEW_DIR / "train.json.gz")
    parser.add_argument(
        "--validation-manifest", type=Path, default=VIEW_DIR / "validation.json.gz"
    )
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments/03_保留专门化/02_论文主线/results/06_Gate开发"
            / "D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"
        ),
    )
    parser.add_argument(
        "--standard-actor",
        type=Path,
        default=(
            PROJECT_ROOT
            / "TD3/pytorch_models"
            / "TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best_actor.pth"
        ),
    )
    parser.add_argument(
        "--interaction-actor",
        type=Path,
        default=(
            PROJECT_ROOT
            / "TD3/pytorch_models"
            / "interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016_actor.pth"
        ),
    )
    parser.add_argument(
        "--a1-main-summary", type=Path, default=A1_MAIN_DIR / "summary.json"
    )
    parser.add_argument(
        "--a1-main-checkpoint", type=Path, default=A1_MAIN_DIR / "any/T1/best.pt"
    )
    parser.add_argument(
        "--student-audit", type=Path, default=B_ROUTE / "local_data/train_audit.json"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=B_ROUTE / "local_data/training/seed20260804",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--temporal-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--temporal-hidden-dim", type=int, default=64)
    parser.add_argument("--max-tracks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def manifest_ids(path, split):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    ids = {item["scenario_id"] for item in payload["scenarios"]}
    if len(ids) != len(payload["scenarios"]):
        raise ValueError("duplicate %s manifest scenario IDs" % split)
    return ids


def verify_dataset_ids(dataset, expected, name):
    observed = set(dataset.scenarios.tolist())
    if observed != expected:
        raise ValueError(
            "%s shard/manifest mismatch: missing=%s extra=%s"
            % (name, sorted(expected - observed)[:5], sorted(observed - expected)[:5])
        )


def concatenate_sources(named_datasets):
    base_features = []
    actor_features = []
    labels = {"front": [], "any": []}
    scenarios = []
    strata = []
    sequence_indices = []
    offset = 0
    for source, dataset in named_datasets:
        count = len(dataset.base_features)
        base_features.append(dataset.base_features)
        actor_features.append(dataset.actor_features)
        for name in labels:
            labels[name].append(dataset.labels[name])
        scenarios.extend(
            ["%s::%s" % (source, scenario) for scenario in dataset.scenarios]
        )
        strata.append(dataset.strata)
        sequence_indices.extend(indices + offset for indices in dataset.sequence_indices)
        offset += count
    result = GateDataset(
        base_features=np.concatenate(base_features),
        actor_features=np.concatenate(actor_features),
        labels={name: np.concatenate(values) for name, values in labels.items()},
        scenarios=np.asarray(scenarios),
        strata=np.concatenate(strata),
        sequence_indices=sequence_indices,
    )
    covered = np.concatenate(result.sequence_indices)
    if len(covered) != offset or set(covered.tolist()) != set(range(offset)):
        raise ValueError("aggregated sequences do not cover each frame exactly once")
    return result


def group_balance_weights(groups):
    groups = np.asarray(groups)
    counts = Counter(groups.tolist())
    weights = np.asarray([1.0 / counts[item] for item in groups], dtype=np.float64)
    return weights / np.sum(weights)


def source_scenario_sample_weights(labels, groups):
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    if labels.shape != groups.shape or labels.ndim != 1:
        raise ValueError("labels and groups must be matching vectors")
    weights = np.zeros(len(labels), dtype=np.float64)
    for group in np.unique(groups):
        group_mask = groups == group
        present = [value for value in (0, 1) if np.any(group_mask & (labels == value))]
        if not present:
            raise ValueError("source-scenario group has no binary labels")
        for value in present:
            mask = group_mask & (labels == value)
            weights[mask] = 1.0 / (len(present) * int(np.sum(mask)))
    if np.any(weights <= 0.0):
        raise ValueError("labels must be binary")
    return (weights / np.mean(weights)).astype(np.float32)


def normalize_source_balanced(train_features, validation_features, groups):
    weights = group_balance_weights(groups)
    mean = np.sum(train_features * weights[:, None], axis=0).astype(np.float32)
    variance = np.sum(((train_features - mean) ** 2) * weights[:, None], axis=0)
    std = np.sqrt(variance).astype(np.float32)
    std[std < 1e-3] = 1.0
    return (
        ((train_features - mean) / std).astype(np.float32),
        ((validation_features - mean) / std).astype(np.float32),
        mean,
        std,
    )


def evaluate_temporal_checkpoint(path, dataset, raw_features, args):
    payload = torch.load(path, map_location=args.device, weights_only=False)
    if payload.get("model_id") != "T1" or payload.get("label") != "any":
        raise ValueError("A1 reference is not an any-label T1 checkpoint")
    if int(payload.get("sequence_length", -1)) != args.sequence_length:
        raise ValueError("A1 reference sequence length mismatch")
    normalized = (
        raw_features - np.asarray(payload["feature_mean"], dtype=np.float32)
    ) / np.asarray(payload["feature_std"], dtype=np.float32)
    windows, indices = make_temporal_windows(
        normalized.astype(np.float32), dataset.sequence_indices, args.sequence_length
    )
    model = TemporalInteractionGate(**payload["model_config"]).to(args.device)
    model.load_state_dict(payload["model_state_dict"])
    ordered = predict(model, windows, args.temporal_batch_size, args.device)
    probabilities = np.zeros(len(dataset.base_features), dtype=np.float32)
    probabilities[indices] = ordered
    return evaluate_probabilities(
        probabilities, dataset, "any", float(payload["threshold"])
    )


def main():
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.experiment_id = "G11-B2"
    args.threshold_policy = "match-s0-fpr"
    args.labels = ["any"]
    if args.device != "cpu" and not torch.cuda.is_available():
        raise ValueError("requested device is unavailable: %s" % args.device)
    if args.epochs < 1 or args.sequence_length < 2 or args.max_tracks < 1:
        raise ValueError("training dimensions must be positive")

    manifest_hashes = {
        "train": sha256_file(args.train_manifest),
        "validation": sha256_file(args.validation_manifest),
    }
    if manifest_hashes != EXPECTED_A1_MANIFEST_SHA256:
        raise ValueError("G11-B2 manifest hash mismatch")
    train_ids = manifest_ids(args.train_manifest, "train")
    validation_ids = manifest_ids(args.validation_manifest, "validation")
    if train_ids & validation_ids:
        raise ValueError("G11-B2 train/validation manifests overlap")
    if sha256_file(args.a1_main_summary) != A1_MAIN_SUMMARY_SHA256:
        raise ValueError("frozen A1 main summary hash mismatch")
    if sha256_file(args.a1_main_checkpoint) != A1_MAIN_T1_SHA256:
        raise ValueError("frozen A1 main T1 hash mismatch")
    student_audit = json.loads(args.student_audit.read_text(encoding="utf-8"))
    if (
        student_audit.get("shards") != 640
        or student_audit.get("dataset_sha256") != G11_B_STUDENT_DATASET_SHA256
    ):
        raise ValueError("G11-B student audit does not match the frozen dataset")

    frozen_hashes = {
        "standard_actor": verify_frozen_actor(
            args.standard_actor, EXPECTED_STANDARD_ACTOR_SHA256
        ),
        "interaction_actor": verify_frozen_actor(
            args.interaction_actor, EXPECTED_INTERACTION_ACTOR_SHA256
        ),
        "detector": verify_frozen_actor(
            args.detector_checkpoint, EXPECTED_DETECTOR_SHA256
        ),
    }
    reset_random_seed(args.seed)
    detector_payload = torch.load(
        args.detector_checkpoint, map_location=args.device, weights_only=False
    )
    detector = LocalRobotDetector(**detector_payload.get("model_config", {})).to(
        args.device
    )
    detector.load_state_dict(detector_payload["model_state_dict"])
    detector.eval()
    standard_actor = load_actor(args.standard_actor, args.device)
    interaction_actor = load_actor(args.interaction_actor, args.device)

    def load(paths, split):
        return build_dataset(
            list_shards(paths),
            detector,
            standard_actor,
            interaction_actor,
            args,
            split,
        )

    a1_train = load(args.a1_train_dir, "train")
    student_train = load(args.student_train_dir, "train")
    validation = load(args.validation_dir, "validation")
    verify_dataset_ids(a1_train, train_ids, "A1 train")
    verify_dataset_ids(student_train, train_ids, "student train")
    verify_dataset_ids(validation, validation_ids, "A1 validation")
    train = concatenate_sources((("a1_5a", a1_train), ("student", student_train)))

    train_raw = np.concatenate((train.base_features, train.actor_features), axis=1)
    validation_raw = np.concatenate(
        (validation.base_features, validation.actor_features), axis=1
    )
    train_normalized, validation_normalized, mean, std = normalize_source_balanced(
        train_raw, validation_raw, train.scenarios
    )
    temporal_train, train_indices = make_temporal_windows(
        train_normalized, train.sequence_indices, args.sequence_length
    )
    temporal_validation, validation_indices = make_temporal_windows(
        validation_normalized, validation.sequence_indices, args.sequence_length
    )

    a1_summary = json.loads(args.a1_main_summary.read_text(encoding="utf-8"))
    s0_metrics = a1_summary["results"]["any"]["S0"]["validation"]
    fpr_caps = {"fpr": s0_metrics["fpr"], "weak_fpr": s0_metrics["weak_fpr"]}
    a1_reference = evaluate_temporal_checkpoint(
        args.a1_main_checkpoint, validation, validation_raw, args
    )
    source_counts = {
        "a1_5a": {
            "scenarios": len(set(a1_train.scenarios.tolist())),
            "frames": len(a1_train.base_features),
        },
        "student": {
            "scenarios": len(set(student_train.scenarios.tolist())),
            "frames": len(student_train.base_features),
            "dataset_sha256": G11_B_STUDENT_DATASET_SHA256,
        },
    }
    args.checkpoint_metadata = {
        "balance_unit": "source+scenario_id",
        "normalization": "source+scenario_id weighted",
        "sources": source_counts,
        "train_manifest_sha256": manifest_hashes["train"],
        "validation_manifest_sha256": manifest_hashes["validation"],
        "a1_reference_checkpoint_sha256": A1_MAIN_T1_SHA256,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = train_candidate(
        "T1",
        TemporalInteractionGate(train_raw.shape[1], args.temporal_hidden_dim),
        temporal_train,
        temporal_validation,
        train,
        validation,
        train_indices,
        validation_indices,
        "any",
        args.temporal_batch_size,
        mean,
        std,
        args,
        fpr_caps,
        source_scenario_sample_weights(train.labels["any"], train.scenarios),
    )
    candidate = result["validation"]
    summary = {
        "protocol": {
            "experiment_id": "G11-B2",
            "sealed_test_read": False,
            "device": args.device,
            "seed": args.seed,
            "epochs": args.epochs,
            "sequence_length": args.sequence_length,
            "balance_unit": "source+scenario_id",
            "normalization": "source+scenario_id weighted",
            "base_feature_dim": gate_feature_dim(24, args.max_tracks),
            "actor_comparison_dim": ACTOR_COMPARISON_DIM,
            "frozen_hashes": frozen_hashes,
            "manifest_hashes": manifest_hashes,
            "a1_main_summary_sha256": A1_MAIN_SUMMARY_SHA256,
            "a1_main_checkpoint_sha256": A1_MAIN_T1_SHA256,
            "fpr_caps": fpr_caps,
        },
        "sources": source_counts,
        "samples": {
            "train_frames": len(train.base_features),
            "train_ego_sequences": len(train.sequence_indices),
            "validation_frames": len(validation.base_features),
            "validation_ego_sequences": len(validation.sequence_indices),
            "train_positive_rate": float(np.mean(train.labels["any"])),
            "validation_positive_rate": float(np.mean(validation.labels["any"])),
        },
        "a1_reference_validation": a1_reference,
        "result": result,
        "validation_delta_vs_a1": {
            "f1": candidate["f1"] - a1_reference["f1"],
            "average_precision": candidate["average_precision"]
            - a1_reference["average_precision"],
            "positive_interval_iou": candidate["events"]["positive_interval_iou"]
            - a1_reference["events"]["positive_interval_iou"],
            "fpr": candidate["fpr"] - a1_reference["fpr"],
            "weak_fpr": candidate["weak_fpr"] - a1_reference["weak_fpr"],
            "total_switches": candidate["events"]["total_switches"]
            - a1_reference["events"]["total_switches"],
        },
        "offline_fpr_admission": bool(
            candidate["fpr"] <= fpr_caps["fpr"] + 1e-12
            and candidate["weak_fpr"] <= fpr_caps["weak_fpr"] + 1e-12
        ),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"summary": str(summary_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
