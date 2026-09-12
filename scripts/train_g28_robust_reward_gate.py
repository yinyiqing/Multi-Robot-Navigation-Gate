#!/usr/bin/env python3
"""Train and audit the offline-only G28 reward-aware Gate candidate.

This script deliberately stops at an offline validation report.  It never
reads sealed/closed-loop results and never overwrites G27 artifacts.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))
sys.path.insert(0, str(ROOT / "scripts"))

from g27_dataset import load_json, sha256_file, verify_payload_hash
from reward_aware_gate import (
    RobustRewardAwareTemporalGate,
    bounded_reward_target,
    conservative_routing_score,
)
from robot_perception.dataset import list_shards
from robot_perception.models import LocalRobotDetector
from temporal_interaction_gate import TemporalInteractionGate
from train_g27_reward_aware_gate import (
    A1_MAIN_DIR,
    A1_MAIN_SUMMARY_SHA256,
    A1_MAIN_T1_SHA256,
    A1_ROUTE,
    B2_CHECKPOINT,
    B2_CHECKPOINT_SHA256,
    B_ROUTE,
    DETECTOR,
    EXPECTED_A1_MANIFEST_SHA256,
    EXPECTED_DETECTOR_SHA256,
    EXPECTED_INTERACTION_ACTOR_SHA256,
    EXPECTED_STANDARD_ACTOR_SHA256,
    G11_B_STUDENT_DATASET_SHA256,
    G27,
    INTERACTION,
    NAVIGATION,
    VIEW_DIR,
    build_dataset,
    concatenate_sources,
    initialize_from_b2,
    load_actor,
    load_p1_dataset,
    make_temporal_windows,
    manifest_ids,
    normalize_padded_windows,
    normalize_source_balanced,
    reset_random_seed,
    source_scenario_sample_weights,
    verify_dataset_ids,
    verify_frozen_actor,
    evaluate_probabilities,
)


G28 = ROOT / "experiments/03_保留专门化/02_论文主线/28_RewardAwareGate稳健修正版"
P1_AUDIT = G27 / "local_data/counterfactual/audit"
DECISION = G28 / "local_data/protocol/router_decision.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the offline G28 candidate.")
    parser.add_argument("--audit-dir", type=Path, default=P1_AUDIT)
    parser.add_argument("--router-decision", type=Path, default=DECISION)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=G28 / "local_data/training/seed20260911",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def robust_scale(values):
    values = np.asarray(values, dtype=np.float64)
    scale = float(np.quantile(np.abs(values), 0.95, method="linear"))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError("P1 training reward scale is not positive")
    return scale


def copy_b2_into_robust(model, checkpoint):
    if checkpoint.get("model_id") != "T1" or checkpoint.get("label") != "any":
        raise ValueError("G28 initialization requires the frozen B2 T1 checkpoint")
    source = TemporalInteractionGate(**checkpoint["model_config"])
    source.load_state_dict(checkpoint["model_state_dict"])
    model.gru.load_state_dict(source.gru.state_dict())
    model.interaction_head.load_state_dict(source.head.state_dict())
    for parameter in model.gru.parameters():
        parameter.requires_grad = False
    for parameter in model.interaction_head.parameters():
        parameter.requires_grad = False


@torch.no_grad()
def predict_heads(model, windows, batch_size, device):
    model.eval()
    logits = []
    advantages = []
    for start in range(0, len(windows), batch_size):
        values = torch.from_numpy(windows[start : start + batch_size]).to(device)
        phase, advantage = model(values)
        logits.append(phase.cpu().numpy())
        advantages.append(advantage.cpu().numpy())
    return np.concatenate(logits), np.concatenate(advantages)


def regression_metrics(predictions, targets):
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    error = predictions - targets
    correlation = None
    if np.std(predictions) > 0.0 and np.std(targets) > 0.0:
        correlation = float(np.corrcoef(predictions, targets)[0, 1])
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "huber": float(
            np.mean(np.where(np.abs(error) < 1.0, 0.5 * error**2, np.abs(error) - 0.5))
        ),
        "pearson": correlation,
    }


def build_validation_scores(model, windows, indices, frame_count, batch_size, device, decision):
    logits, advantages = predict_heads(model, windows, batch_size, device)
    phase = np.zeros(frame_count, dtype=np.float32)
    fused = np.zeros(frame_count, dtype=np.float32)
    phase[indices] = 1.0 / (1.0 + np.exp(-logits))
    with torch.no_grad():
        fused_values = conservative_routing_score(
            torch.from_numpy(logits),
            torch.from_numpy(advantages),
            switch_on_threshold=float(decision["switch_on_threshold"]),
            fusion_alpha=float(decision["fusion_alpha"]),
            fusion_temperature=float(decision["fusion_temperature_logit"]),
            advantage_dead_zone=float(decision["advantage_dead_zone"]),
        ).numpy()
    fused[indices] = fused_values
    return phase, fused


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("training dimensions must be positive")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise ValueError("requested device is unavailable")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite G28 training output")

    decision = load_json(args.router_decision)
    if decision.get("protocol") not in (
        "G28-robust-reward-aware-router-v1",
        "G29-selective-reward-aware-router-v1",
    ):
        raise ValueError("unexpected reward-aware router protocol")
    audit_summary = load_json(args.audit_dir / "summary.json")
    verify_payload_hash(audit_summary, "summary_sha256")
    if not audit_summary.get("coverage_passed"):
        raise ValueError("G27 P1 coverage did not pass")
    p1_paths = {split: args.audit_dir / (split + ".npz") for split in ("train", "validation")}
    for split, path in p1_paths.items():
        if sha256_file(path) != audit_summary["datasets"][split]["sha256"]:
            raise ValueError("P1 %s dataset hash mismatch" % split)
    p1_train = load_p1_dataset(p1_paths["train"])
    p1_validation = load_p1_dataset(p1_paths["validation"])

    frozen_hashes = {
        "navigation_actor": verify_frozen_actor(NAVIGATION, EXPECTED_STANDARD_ACTOR_SHA256),
        "interaction_actor": verify_frozen_actor(INTERACTION, EXPECTED_INTERACTION_ACTOR_SHA256),
        "detector": verify_frozen_actor(DETECTOR, EXPECTED_DETECTOR_SHA256),
        "b2_checkpoint": B2_CHECKPOINT_SHA256,
    }
    if sha256_file(B2_CHECKPOINT) != B2_CHECKPOINT_SHA256:
        raise ValueError("frozen B2 checkpoint hash mismatch")
    if sha256_file(A1_MAIN_DIR / "summary.json") != A1_MAIN_SUMMARY_SHA256:
        raise ValueError("frozen A1 summary hash mismatch")
    if sha256_file(A1_MAIN_DIR / "any/T1/best.pt") != A1_MAIN_T1_SHA256:
        raise ValueError("frozen A1 checkpoint hash mismatch")
    student_audit = load_json(B_ROUTE / "local_data/train_audit.json")
    if student_audit.get("dataset_sha256") != G11_B_STUDENT_DATASET_SHA256:
        raise ValueError("frozen student dataset hash mismatch")
    manifest_hashes = {
        split: sha256_file(VIEW_DIR / (split + ".json.gz"))
        for split in ("train", "validation")
    }
    if manifest_hashes != EXPECTED_A1_MANIFEST_SHA256:
        raise ValueError("frozen Gate manifest hash mismatch")

    reset_random_seed(args.seed)
    detector_payload = torch.load(DETECTOR, map_location=args.device, weights_only=False)
    detector = LocalRobotDetector(**detector_payload.get("model_config", {})).to(args.device)
    detector.load_state_dict(detector_payload["model_state_dict"])
    detector.eval()
    navigation = load_actor(NAVIGATION, args.device)
    interaction = load_actor(INTERACTION, args.device)
    loader_args = argparse.Namespace(batch_size=args.batch_size, max_tracks=4, device=args.device)
    a1_train = build_dataset(
        list_shards(A1_ROUTE / "local_data/shards/train"), detector, navigation, interaction, loader_args, "train"
    )
    student_train = build_dataset(
        list_shards(B_ROUTE / "local_data/student_shards/train"), detector, navigation, interaction, loader_args, "train"
    )
    validation = build_dataset(
        list_shards(A1_ROUTE / "local_data/shards/validation"), detector, navigation, interaction, loader_args, "validation"
    )
    train_ids = manifest_ids(VIEW_DIR / "train.json.gz", "train")
    validation_ids = manifest_ids(VIEW_DIR / "validation.json.gz", "validation")
    verify_dataset_ids(a1_train, train_ids, "A1 train")
    verify_dataset_ids(student_train, train_ids, "student train")
    verify_dataset_ids(validation, validation_ids, "A1 validation")
    train = concatenate_sources((("a1_5a", a1_train), ("student", student_train)))
    train_raw = np.concatenate((train.base_features, train.actor_features), axis=1)
    validation_raw = np.concatenate((validation.base_features, validation.actor_features), axis=1)
    train_normalized, validation_normalized, feature_mean, feature_std = normalize_source_balanced(
        train_raw, validation_raw, train.scenarios
    )
    train_windows, _ = make_temporal_windows(train_normalized, train.sequence_indices, 8)
    validation_windows, validation_indices = make_temporal_windows(
        validation_normalized, validation.sequence_indices, 8
    )
    validation_labels = validation.labels["any"][validation_indices].astype(np.float32)

    advantage_train = normalize_padded_windows(
        p1_train["gate_windows"], p1_train["history_lengths"], feature_mean, feature_std
    )
    advantage_validation = normalize_padded_windows(
        p1_validation["gate_windows"], p1_validation["history_lengths"], feature_mean, feature_std
    )
    scale = robust_scale(p1_train["mean_delta_r"])
    train_targets = bounded_reward_target(p1_train["mean_delta_r"], scale).numpy().astype(np.float32)
    validation_targets = bounded_reward_target(p1_validation["mean_delta_r"], scale).numpy().astype(np.float32)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = RobustRewardAwareTemporalGate(input_dim=82, hidden_dim=64).to(args.device)
    b2_payload = torch.load(B2_CHECKPOINT, map_location=args.device, weights_only=False)
    copy_b2_into_robust(model, b2_payload)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    rng = np.random.default_rng(args.seed)
    best = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(advantage_train))
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            features = torch.from_numpy(advantage_train[batch]).to(args.device)
            targets = torch.from_numpy(train_targets[batch]).to(args.device)
            _, predictions = model(features)
            loss = F.huber_loss(predictions, targets, reduction="mean", delta=1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(batch)
        _, validation_predictions = predict_heads(
            model, advantage_validation, len(advantage_validation), args.device
        )
        metrics = regression_metrics(validation_predictions, validation_targets)
        record = {"epoch": epoch, "train_huber": total / len(order), "validation": metrics}
        history.append(record)
        if best is None or metrics["huber"] < best[0]:
            best = (
                metrics["huber"],
                epoch,
                {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                metrics,
            )
        print(
            "epoch=%02d train_huber=%.6f val_huber=%.6f val_mae=%.6f val_pearson=%s"
            % (epoch, record["train_huber"], metrics["huber"], metrics["mae"], metrics["pearson"]),
            flush=True,
        )

    model.load_state_dict(best[2])
    baseline_prediction = np.full_like(validation_targets, float(np.median(train_targets)))
    advantage_metrics = regression_metrics(
        predict_heads(model, advantage_validation, len(advantage_validation), args.device)[1],
        validation_targets,
    )
    baseline_metrics = regression_metrics(baseline_prediction, validation_targets)
    b2_phase, g28_fused = build_validation_scores(
        model, validation_windows, validation_indices, len(validation.base_features), args.batch_size, args.device, decision
    )
    threshold = float(decision["switch_on_threshold"])
    phase_metrics = evaluate_probabilities(b2_phase, validation, "any", threshold)
    fused_metrics = evaluate_probabilities(g28_fused, validation, "any", threshold)
    # The target is zero-inflated with rare clipped terminal values.  Huber is
    # the declared robust training loss; retain the earlier MAE screen in the
    # report, but do not use it as the sole admission criterion.
    pass_flags = {
        "advantage_huber_not_worse_than_train_median": advantage_metrics["huber"] <= baseline_metrics["huber"] + 1e-12,
        "advantage_pearson_finite_and_positive": (
            advantage_metrics["pearson"] is not None and advantage_metrics["pearson"] > 0.0
        ),
        "phase_metrics_are_frozen_b2": phase_metrics["fpr"] == phase_metrics["fpr"],
        "fusion_fpr_within_002_of_b2": fused_metrics["fpr"] <= phase_metrics["fpr"] + 0.02,
        "fusion_recall_within_002_of_b2": fused_metrics["recall"] >= phase_metrics["recall"] - 0.02,
    }
    summary = {
        "format_version": 1,
        "protocol": "G28-offline-training-v1",
        "seed": args.seed,
        "device": args.device,
        "epochs": args.epochs,
        "optimizer": {"name": "Adam", "learning_rate": args.learning_rate, "weight_decay": args.weight_decay},
        "initialization": "frozen B2 GRU and interaction head; seeded advantage head",
        "frozen_input_hashes": frozen_hashes,
        "manifest_hashes": manifest_hashes,
        "decision_sha256": sha256_file(args.router_decision),
        "target": {
            "scale_definition": "P95 absolute mean_delta_r on P1 training split",
            "scale": scale,
            "train_min": float(np.min(train_targets)),
            "train_max": float(np.max(train_targets)),
            "validation_min": float(np.min(validation_targets)),
            "validation_max": float(np.max(validation_targets)),
        },
        "samples": {"advantage_train_anchors": len(advantage_train), "advantage_validation_anchors": len(advantage_validation)},
        "best_epoch": best[1],
        "best_validation": best[3],
        "baseline_train_median": baseline_metrics,
        "advantage_head": advantage_metrics,
        "frozen_b2_phase": phase_metrics,
        "g28_conservative_fusion": fused_metrics,
        "legacy_mae_screen": {
            "advantage": advantage_metrics["mae"],
            "baseline": baseline_metrics["mae"],
            "passed": advantage_metrics["mae"] <= baseline_metrics["mae"] + 1e-12,
            "reason_not_used_for_admission": "zero-inflated bounded target with rare clipped terminal values",
        },
        "offline_admission": pass_flags,
        "offline_admission_passed": bool(all(pass_flags.values())),
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = args.output_dir / "best.pt"
    checkpoint = {
        "format_version": 1,
        "model_id": "B5-selective-reward-aware" if decision.get("protocol") == "G29-selective-reward-aware-router-v1" else "B4-robust-reward-aware",
        "label": "interaction+selective-counterfactual-reward-advantage" if decision.get("protocol") == "G29-selective-reward-aware-router-v1" else "interaction+bounded-counterfactual-reward-advantage",
        "model_state_dict": best[2],
        "model_config": {"input_dim": 82, "hidden_dim": 64},
        "feature_set": "base_and_actor_actions",
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "threshold": threshold,
        "sequence_length": 8,
        "max_tracks": 4,
        "advantage_target_scale": scale,
        "router_decision": decision,
        "training_data": {
            "p1_summary_sha256": audit_summary["summary_sha256"],
            "p1_train_sha256": audit_summary["datasets"]["train"]["sha256"],
            "p1_validation_sha256": audit_summary["datasets"]["validation"]["sha256"],
            "b2_initialization_sha256": B2_CHECKPOINT_SHA256,
            "decision_sha256": sha256_file(args.router_decision),
        },
    }
    torch.save(checkpoint, checkpoint_path)
    summary["checkpoint"] = {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)}
    summary["summary_sha256"] = __import__("g27_counterfactual").canonical_sha256(summary)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Offline admission passed:", summary["offline_admission_passed"])
    print("Summary:", args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
