#!/usr/bin/env python3
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

from g27_counterfactual import canonical_sha256
from g27_dataset import load_json, sha256_file, verify_payload_hash
from reward_aware_gate import RewardAwareTemporalGate, combined_routing_score
from robot_perception.dataset import list_shards
from robot_perception.models import LocalRobotDetector
from temporal_interaction_gate import TemporalInteractionGate
from train_g11_b_aggregated_gate import (
    A1_MAIN_DIR,
    A1_MAIN_SUMMARY_SHA256,
    A1_MAIN_T1_SHA256,
    A1_ROUTE,
    B_ROUTE,
    G11_B_STUDENT_DATASET_SHA256,
    VIEW_DIR,
    concatenate_sources,
    manifest_ids,
    normalize_source_balanced,
    source_scenario_sample_weights,
    verify_dataset_ids,
)
from train_temporal_interaction_gate import (
    EXPECTED_A1_MANIFEST_SHA256,
    EXPECTED_DETECTOR_SHA256,
    EXPECTED_INTERACTION_ACTOR_SHA256,
    EXPECTED_STANDARD_ACTOR_SHA256,
    build_dataset,
    evaluate_probabilities,
    load_actor,
    make_temporal_windows,
    reset_random_seed,
    verify_frozen_actor,
)


G27 = (
    ROOT
    / "experiments/03_保留专门化/02_论文主线/27_反事实Reward增强Gate监督"
)
DETECTOR = (
    ROOT
    / "experiments/03_保留专门化/02_论文主线/results/06_Gate开发"
    / "D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"
)
NAVIGATION = (
    ROOT
    / "TD3/pytorch_models"
    / "TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best_actor.pth"
)
INTERACTION = (
    ROOT
    / "TD3/pytorch_models"
    / "interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016_actor.pth"
)
B2_CHECKPOINT = B_ROUTE / "local_data/training/seed20260804/any/T1/best.pt"
B2_CHECKPOINT_SHA256 = "fc59b4f783f7c5461ebb0239fab4b34896ad910ee78e7223e88d29ce9c3f5a52"
ROUTER_DECISION_SHA256 = "0abeed87aefaf38253449b563b98bcdef46ab4160b994e2c7853ed5836abc11d"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the single G27 B3 Router.")
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=G27 / "local_data/counterfactual/audit",
    )
    parser.add_argument(
        "--router-decision",
        type=Path,
        default=G27 / "local_data/protocol/router_decision.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=G27 / "local_data/training/seed20260910",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def normalize_padded_windows(windows, history_lengths, mean, std):
    windows = np.asarray(windows, dtype=np.float32)
    history_lengths = np.asarray(history_lengths, dtype=np.int64)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if windows.ndim != 3 or windows.shape[1:] != (8, 82):
        raise ValueError("P1 Gate windows must have shape [N,8,82]")
    if history_lengths.shape != (len(windows),):
        raise ValueError("P1 history lengths do not match Gate windows")
    if np.any(history_lengths < 1) or np.any(history_lengths > 8):
        raise ValueError("P1 history lengths must lie in [1,8]")
    if mean.shape != (82,) or std.shape != (82,) or np.any(std <= 0.0):
        raise ValueError("feature normalization must contain 82 positive scales")
    normalized = np.zeros_like(windows)
    for index, length in enumerate(history_lengths):
        normalized[index, -length:] = (windows[index, -length:] - mean) / std
    return normalized


def load_p1_dataset(path):
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "anchor_ids",
            "scenario_ids",
            "strata",
            "history_lengths",
            "gate_windows",
            "interaction_labels",
            "delta_r_1",
            "delta_r_2",
            "mean_delta_r",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError("P1 dataset is missing %s" % missing)
        return {name: payload[name].copy() for name in required}


def initialize_from_b2(model, checkpoint):
    if checkpoint.get("model_id") != "T1" or checkpoint.get("label") != "any":
        raise ValueError("B3 initialization requires the frozen B2 T1 checkpoint")
    source = TemporalInteractionGate(**checkpoint["model_config"])
    source.load_state_dict(checkpoint["model_state_dict"])
    model.gru.load_state_dict(source.gru.state_dict())
    model.interaction_head.load_state_dict(source.head.state_dict())


@torch.no_grad()
def predict_heads(model, features, batch_size, device):
    model.eval()
    interaction = []
    advantage = []
    for start in range(0, len(features), batch_size):
        values = torch.from_numpy(features[start : start + batch_size]).to(device)
        logits, predictions = model(values)
        interaction.append(logits.cpu().numpy())
        advantage.append(predictions.cpu().numpy())
    return np.concatenate(interaction), np.concatenate(advantage)


def regression_metrics(predictions, targets):
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    error = predictions - targets
    correlation = None
    if len(targets) > 1 and np.std(targets) > 0.0 and np.std(predictions) > 0.0:
        correlation = float(np.corrcoef(predictions, targets)[0, 1])
    nonzero = targets != 0.0
    return {
        "huber": float(np.mean(np.where(np.abs(error) < 1.0, 0.5 * error**2, np.abs(error) - 0.5))),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "pearson": correlation,
        "sign_accuracy": float(np.mean(np.sign(predictions[nonzero]) == np.sign(targets[nonzero])))
        if np.any(nonzero)
        else None,
    }


def interaction_probabilities(model, windows, target_indices, frame_count, batch_size, device):
    logits, advantages = predict_heads(model, windows, batch_size, device)
    probabilities = np.zeros(frame_count, dtype=np.float32)
    combined = np.zeros(frame_count, dtype=np.float32)
    probabilities[target_indices] = 1.0 / (1.0 + np.exp(-logits))
    clipped = np.clip(advantages, -1.0, 1.0)
    combined[target_indices] = 1.0 / (1.0 + np.exp(-(logits + clipped)))
    return probabilities, combined


def evaluate_model(
    model,
    validation_windows,
    validation_indices,
    validation_dataset,
    advantage_windows,
    advantage_targets,
    batch_size,
    device,
    threshold,
):
    interaction_logits, _ = predict_heads(
        model, validation_windows, batch_size, device
    )
    interaction_labels = validation_dataset.labels["any"][validation_indices]
    interaction_loss = float(
        F.binary_cross_entropy_with_logits(
            torch.from_numpy(interaction_logits),
            torch.from_numpy(interaction_labels.astype(np.float32)),
        ).item()
    )
    _, advantage_predictions = predict_heads(
        model, advantage_windows, batch_size, device
    )
    advantage = regression_metrics(advantage_predictions, advantage_targets)
    probabilities, combined = interaction_probabilities(
        model,
        validation_windows,
        validation_indices,
        len(validation_dataset.base_features),
        batch_size,
        device,
    )
    return {
        "joint_loss": 0.5 * interaction_loss + 0.5 * advantage["huber"],
        "interaction_bce": interaction_loss,
        "interaction_head": evaluate_probabilities(
            probabilities, validation_dataset, "any", threshold
        ),
        "combined_score": evaluate_probabilities(
            combined, validation_dataset, "any", threshold
        ),
        "advantage_head": advantage,
    }


def accumulate_epoch(
    model,
    interaction_windows,
    interaction_labels,
    interaction_weights,
    advantage_windows,
    advantage_targets,
    optimizer,
    batch_size,
    rng,
    device,
):
    model.train()
    optimizer.zero_grad()
    interaction_order = rng.permutation(len(interaction_windows))
    weight_total = float(np.sum(interaction_weights))
    interaction_loss = 0.0
    for start in range(0, len(interaction_order), batch_size):
        batch = interaction_order[start : start + batch_size]
        features = torch.from_numpy(interaction_windows[batch]).to(device)
        labels = torch.from_numpy(interaction_labels[batch]).to(device)
        weights = torch.from_numpy(interaction_weights[batch]).to(device)
        logits, _ = model(features)
        loss_sum = torch.sum(F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * weights)
        interaction_loss += float(loss_sum.detach().cpu())
        (0.5 * loss_sum / weight_total).backward()

    advantage_order = rng.permutation(len(advantage_windows))
    advantage_loss = 0.0
    for start in range(0, len(advantage_order), batch_size):
        batch = advantage_order[start : start + batch_size]
        features = torch.from_numpy(advantage_windows[batch]).to(device)
        targets = torch.from_numpy(advantage_targets[batch]).to(device)
        _, predictions = model(features)
        loss_sum = F.huber_loss(predictions, targets, reduction="sum", delta=1.0)
        advantage_loss += float(loss_sum.detach().cpu())
        (0.5 * loss_sum / len(advantage_windows)).backward()
    optimizer.step()
    return {
        "joint_loss": 0.5 * interaction_loss / weight_total
        + 0.5 * advantage_loss / len(advantage_windows),
        "interaction_bce": interaction_loss / weight_total,
        "advantage_huber": advantage_loss / len(advantage_windows),
    }


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("training dimensions must be positive")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise ValueError("requested device is unavailable")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite B3 training output")

    audit_summary_path = args.audit_dir / "summary.json"
    audit_summary = load_json(audit_summary_path)
    verify_payload_hash(audit_summary, "summary_sha256")
    if not audit_summary.get("coverage_passed"):
        raise ValueError("G27 P1 coverage did not pass")
    p1_paths = {
        split: args.audit_dir / (split + ".npz")
        for split in ("train", "validation")
    }
    for split, path in p1_paths.items():
        if sha256_file(path) != audit_summary["datasets"][split]["sha256"]:
            raise ValueError("G27 P1 %s dataset hash mismatch" % split)
    p1_train = load_p1_dataset(p1_paths["train"])
    p1_validation = load_p1_dataset(p1_paths["validation"])

    if sha256_file(args.router_decision) != ROUTER_DECISION_SHA256:
        raise ValueError("frozen Router decision hash mismatch")
    decision = load_json(args.router_decision)
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
    train_ids = manifest_ids(VIEW_DIR / "train.json.gz", "train")
    validation_ids = manifest_ids(VIEW_DIR / "validation.json.gz", "validation")
    if train_ids & validation_ids:
        raise ValueError("Gate train/validation scenes overlap")

    frozen_hashes = {
        "navigation_actor": verify_frozen_actor(NAVIGATION, EXPECTED_STANDARD_ACTOR_SHA256),
        "interaction_actor": verify_frozen_actor(INTERACTION, EXPECTED_INTERACTION_ACTOR_SHA256),
        "detector": verify_frozen_actor(DETECTOR, EXPECTED_DETECTOR_SHA256),
        "b2_checkpoint": B2_CHECKPOINT_SHA256,
    }
    reset_random_seed(args.seed)
    detector_payload = torch.load(DETECTOR, map_location=args.device, weights_only=False)
    detector = LocalRobotDetector(**detector_payload.get("model_config", {})).to(args.device)
    detector.load_state_dict(detector_payload["model_state_dict"])
    detector.eval()
    navigation = load_actor(NAVIGATION, args.device)
    interaction = load_actor(INTERACTION, args.device)

    loader_args = argparse.Namespace(batch_size=args.batch_size, max_tracks=4, device=args.device)
    a1_train = build_dataset(
        list_shards(A1_ROUTE / "local_data/shards/train"),
        detector,
        navigation,
        interaction,
        loader_args,
        "train",
    )
    student_train = build_dataset(
        list_shards(B_ROUTE / "local_data/student_shards/train"),
        detector,
        navigation,
        interaction,
        loader_args,
        "train",
    )
    validation = build_dataset(
        list_shards(A1_ROUTE / "local_data/shards/validation"),
        detector,
        navigation,
        interaction,
        loader_args,
        "validation",
    )
    verify_dataset_ids(a1_train, train_ids, "A1 train")
    verify_dataset_ids(student_train, train_ids, "student train")
    verify_dataset_ids(validation, validation_ids, "A1 validation")
    train = concatenate_sources((("a1_5a", a1_train), ("student", student_train)))

    train_raw = np.concatenate((train.base_features, train.actor_features), axis=1)
    validation_raw = np.concatenate(
        (validation.base_features, validation.actor_features), axis=1
    )
    train_normalized, validation_normalized, feature_mean, feature_std = (
        normalize_source_balanced(train_raw, validation_raw, train.scenarios)
    )
    interaction_train, interaction_train_indices = make_temporal_windows(
        train_normalized, train.sequence_indices, 8
    )
    interaction_validation, interaction_validation_indices = make_temporal_windows(
        validation_normalized, validation.sequence_indices, 8
    )
    interaction_labels = train.labels["any"][interaction_train_indices].astype(np.float32)
    interaction_weights = source_scenario_sample_weights(
        train.labels["any"], train.scenarios
    )[interaction_train_indices]

    advantage_train = normalize_padded_windows(
        p1_train["gate_windows"],
        p1_train["history_lengths"],
        feature_mean,
        feature_std,
    )
    advantage_validation = normalize_padded_windows(
        p1_validation["gate_windows"],
        p1_validation["history_lengths"],
        feature_mean,
        feature_std,
    )
    advantage_mean = float(np.mean(p1_train["mean_delta_r"]))
    advantage_std = float(np.std(p1_train["mean_delta_r"]))
    if not np.isfinite(advantage_std) or advantage_std < 1e-6:
        raise ValueError("P1 training reward advantage has insufficient variance")
    advantage_train_targets = (
        (p1_train["mean_delta_r"] - advantage_mean) / advantage_std
    ).astype(np.float32)
    advantage_validation_targets = (
        (p1_validation["mean_delta_r"] - advantage_mean) / advantage_std
    ).astype(np.float32)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = RewardAwareTemporalGate(input_dim=82, hidden_dim=64).to(args.device)
    b2_payload = torch.load(B2_CHECKPOINT, map_location=args.device, weights_only=False)
    initialize_from_b2(model, b2_payload)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng = np.random.default_rng(args.seed)
    threshold = float(decision["switch_on_threshold"])
    best = None
    history = []
    for epoch in range(1, args.epochs + 1):
        train_losses = accumulate_epoch(
            model,
            interaction_train,
            interaction_labels,
            interaction_weights,
            advantage_train,
            advantage_train_targets,
            optimizer,
            args.batch_size,
            rng,
            args.device,
        )
        validation_metrics = evaluate_model(
            model,
            interaction_validation,
            interaction_validation_indices,
            validation,
            advantage_validation,
            advantage_validation_targets,
            args.batch_size,
            args.device,
            threshold,
        )
        record = {
            "epoch": epoch,
            "train": train_losses,
            "validation": validation_metrics,
        }
        history.append(record)
        key = float(validation_metrics["joint_loss"])
        if best is None or key < best[0]:
            best = (
                key,
                epoch,
                {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                validation_metrics,
            )
        print(
            "epoch=%02d train=%.5f val=%.5f interaction_f1=%.3f advantage_mae=%.3f"
            % (
                epoch,
                train_losses["joint_loss"],
                validation_metrics["joint_loss"],
                validation_metrics["combined_score"]["f1"],
                validation_metrics["advantage_head"]["mae"],
            ),
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = args.output_dir / "best.pt"
    checkpoint = {
        "format_version": 1,
        "model_id": "B3-reward-aware",
        "label": "interaction+counterfactual_reward_advantage",
        "model_state_dict": best[2],
        "model_config": {"input_dim": 82, "hidden_dim": 64},
        "feature_set": "base_and_actor_actions",
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "threshold": threshold,
        "sequence_length": 8,
        "max_tracks": 4,
        "advantage_target_mean": advantage_mean,
        "advantage_target_std": advantage_std,
        "router_decision": decision,
        "training_data": {
            "p1_audit_summary_sha256": audit_summary["summary_sha256"],
            "p1_train_sha256": audit_summary["datasets"]["train"]["sha256"],
            "p1_validation_sha256": audit_summary["datasets"]["validation"]["sha256"],
            "b2_initialization_sha256": B2_CHECKPOINT_SHA256,
            "router_decision_sha256": ROUTER_DECISION_SHA256,
        },
    }
    torch.save(checkpoint, checkpoint_path)
    summary = {
        "format_version": 1,
        "protocol": "G27-B3-training-v1",
        "seed": args.seed,
        "device": args.device,
        "epochs": args.epochs,
        "optimizer": {
            "name": "Adam",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "updates_per_epoch": 1,
        },
        "loss": "0.5 * source-scenario-weighted BCE + 0.5 * masked Huber(delta=1)",
        "initialization": "frozen B2 GRU and interaction head; seeded new advantage head",
        "frozen_input_hashes": frozen_hashes,
        "manifest_hashes": manifest_hashes,
        "router_decision_sha256": ROUTER_DECISION_SHA256,
        "samples": {
            "interaction_train_frames": len(interaction_train),
            "interaction_validation_frames": len(interaction_validation),
            "advantage_train_anchors": len(advantage_train),
            "advantage_validation_anchors": len(advantage_validation),
        },
        "normalization": {
            "features": "source+scenario weighted on original interaction training frames",
            "advantage_mean_train_only": advantage_mean,
            "advantage_std_train_only": advantage_std,
        },
        "best_epoch": best[1],
        "best_validation": best[3],
        "history": history,
        "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Summary:", args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
