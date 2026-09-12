#!/usr/bin/env python3
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))
sys.path.insert(0, str(ROOT / "scripts"))

from g27_counterfactual import canonical_sha256
from g27_dataset import sha256_file
from reward_aware_gate import (
    DualEncoderRewardTemporalGate,
    bounded_reward_target,
    joint_supervision_routing_score,
)
from robot_perception.dataset import list_shards
from robot_perception.models import LocalRobotDetector
from train_g11_b_aggregated_gate import (
    A1_ROUTE,
    B_ROUTE,
    VIEW_DIR,
    concatenate_sources,
    manifest_ids,
    normalize_source_balanced,
    source_scenario_sample_weights,
    verify_dataset_ids,
)
from train_g27_reward_aware_gate import normalize_padded_windows
from train_temporal_interaction_gate import (
    build_dataset,
    evaluate_probabilities,
    load_actor,
    make_temporal_windows,
    reset_random_seed,
)


BASE = ROOT / "experiments/03_保留专门化/02_论文主线"
G33 = BASE / "33_双头联合监督预检/local_data"
G34 = BASE / "34_双头RewardAwareGate"
DETECTOR = (
    BASE
    / "results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"
)
NAVIGATION = (
    ROOT
    / "TD3/pytorch_models/TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best_actor.pth"
)
INTERACTION = (
    ROOT
    / "TD3/pytorch_models/interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016_actor.pth"
)
DEFAULT_REWARD_TRAIN = [
    BASE / "27_反事实Reward增强Gate监督/local_data/counterfactual/audit/train.npz",
    G33 / "counterfactual/audit/train.npz",
    G33 / "counterfactual_round2/audit/train.npz",
]
DEFAULT_REWARD_VALIDATION = [
    BASE / "27_反事实Reward增强Gate监督/local_data/counterfactual/audit/validation.npz",
    G33 / "counterfactual/audit/validation.npz",
    G33 / "counterfactual_round2/audit/validation.npz",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=G34 / "local_data/training/seed20260912"
    )
    parser.add_argument("--phase-epochs", type=int, default=40)
    parser.add_argument("--reward-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_reward_sources(paths):
    required = {
        "anchor_ids",
        "scenario_ids",
        "gate_windows",
        "history_lengths",
        "interaction_labels",
        "mean_delta_r",
    }
    parts = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            missing = sorted(required - set(data.files))
            if missing:
                raise ValueError("%s is missing %s" % (path, missing))
            part = {name: data[name].copy() for name in required}
        if part["gate_windows"].shape[1:] != (8, 82):
            raise ValueError("reward Gate windows must have shape [N,8,82]")
        count = len(part["anchor_ids"])
        if any(len(values) != count for values in part.values()):
            raise ValueError("reward source arrays have different lengths")
        parts.append(part)
    return {
        name: np.concatenate([part[name] for part in parts], axis=0)
        for name in required
    }


def scenario_weights(scenarios):
    counts = Counter(str(value) for value in scenarios)
    weights = np.asarray([1.0 / counts[str(value)] for value in scenarios], dtype=np.float32)
    return weights / np.mean(weights)


def robust_scale_by_anchor(anchor_ids, delta):
    grouped = {}
    for anchor, value in zip(anchor_ids, delta):
        grouped.setdefault(str(anchor), []).append(float(value))
    representative = np.asarray(
        [np.median(values) for values in grouped.values()], dtype=np.float64
    )
    scale = float(np.percentile(np.abs(representative), 95.0))
    if not np.isfinite(scale) or scale <= 1e-6:
        raise ValueError("reward differences have insufficient robust scale")
    return scale, len(grouped)


@torch.no_grad()
def predict(model, features, batch_size, device):
    model.eval()
    phase = []
    reward = []
    for start in range(0, len(features), batch_size):
        values = torch.from_numpy(features[start : start + batch_size]).to(device)
        phase_logits, reward_values = model(values)
        phase.append(phase_logits.cpu().numpy())
        reward.append(reward_values.cpu().numpy())
    return np.concatenate(phase), np.concatenate(reward)


def weighted_huber(prediction, target, weights):
    error = np.abs(np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64))
    loss = np.where(error < 1.0, 0.5 * error**2, error - 0.5)
    return float(np.average(loss, weights=np.asarray(weights, dtype=np.float64)))


def reward_metrics(prediction, target, weights, raw_delta):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    correlation = None
    if np.std(prediction) > 0.0 and np.std(target) > 0.0:
        correlation = float(np.corrcoef(prediction, target)[0, 1])
    reliable = np.abs(np.asarray(raw_delta)) > 0.005
    return {
        "weighted_huber": weighted_huber(prediction, target, weights),
        "mae": float(np.mean(np.abs(prediction - target))),
        "pearson": correlation,
        "reliable_count": int(reliable.sum()),
        "reliable_sign_accuracy": (
            float(np.mean(np.sign(prediction[reliable]) == np.sign(target[reliable])))
            if reliable.any()
            else None
        ),
    }


def phase_metrics(model, windows, indices, dataset, batch_size, device, reward_weight):
    logits, reward = predict(model, windows, batch_size, device)
    base = torch.sigmoid(torch.from_numpy(logits)).numpy()
    joint = joint_supervision_routing_score(
        torch.from_numpy(logits), torch.from_numpy(reward), reward_weight=reward_weight
    ).numpy()
    base_full = np.zeros(len(dataset.base_features), dtype=np.float32)
    joint_full = np.zeros(len(dataset.base_features), dtype=np.float32)
    base_full[indices] = base
    joint_full[indices] = joint
    return (
        evaluate_probabilities(base_full, dataset, "any", 0.43),
        evaluate_probabilities(joint_full, dataset, "any", 0.43),
    )


def train_phase(model, train_x, train_y, train_w, validation_x, validation_y, args):
    parameters = list(model.interaction_gru.parameters()) + list(model.interaction_head.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    best = None
    history = []
    for epoch in range(1, args.phase_epochs + 1):
        model.train()
        order = rng.permutation(len(train_x))
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            x = torch.from_numpy(train_x[batch]).to(args.device)
            y = torch.from_numpy(train_y[batch]).to(args.device)
            w = torch.from_numpy(train_w[batch]).to(args.device)
            logits, _ = model(x)
            loss = torch.sum(F.binary_cross_entropy_with_logits(logits, y, reduction="none") * w) / torch.sum(w)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        logits, _ = predict(model, validation_x, args.batch_size, args.device)
        value = float(F.binary_cross_entropy_with_logits(
            torch.from_numpy(logits), torch.from_numpy(validation_y)
        ).item())
        history.append({"epoch": epoch, "train_bce": float(np.mean(losses)), "validation_bce": value})
        if best is None or value < best[0]:
            best = (
                value,
                epoch,
                {name: tensor.detach().cpu().clone() for name, tensor in model.interaction_gru.state_dict().items()},
                {name: tensor.detach().cpu().clone() for name, tensor in model.interaction_head.state_dict().items()},
            )
    model.interaction_gru.load_state_dict(best[2])
    model.interaction_head.load_state_dict(best[3])
    return best[1], history


def train_reward(model, train_x, train_y, train_w, validation_x, validation_y, validation_w, validation_delta, args):
    parameters = list(model.reward_gru.parameters()) + list(model.reward_head.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed + 1)
    best = None
    history = []
    for epoch in range(1, args.reward_epochs + 1):
        model.train()
        order = rng.permutation(len(train_x))
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            x = torch.from_numpy(train_x[batch]).to(args.device)
            y = torch.from_numpy(train_y[batch]).to(args.device)
            w = torch.from_numpy(train_w[batch]).to(args.device)
            _, prediction = model(x)
            per_item = F.huber_loss(prediction, y, reduction="none", delta=1.0)
            loss = torch.sum(per_item * w) / torch.sum(w)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        _, prediction = predict(model, validation_x, args.batch_size, args.device)
        metrics = reward_metrics(prediction, validation_y, validation_w, validation_delta)
        history.append({"epoch": epoch, "train_huber": float(np.mean(losses)), "validation": metrics})
        value = metrics["weighted_huber"]
        if best is None or value < best[0]:
            best = (
                value,
                epoch,
                {name: tensor.detach().cpu().clone() for name, tensor in model.reward_gru.state_dict().items()},
                {name: tensor.detach().cpu().clone() for name, tensor in model.reward_head.state_dict().items()},
                metrics,
            )
    model.reward_gru.load_state_dict(best[2])
    model.reward_head.load_state_dict(best[3])
    return best[1], best[4], history


def main():
    args = parse_args()
    if args.device != "cpu" and not torch.cuda.is_available():
        raise ValueError("requested device is unavailable")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite G34 output")
    all_inputs = [DETECTOR, NAVIGATION, INTERACTION] + DEFAULT_REWARD_TRAIN + DEFAULT_REWARD_VALIDATION
    for path in all_inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    train_ids = manifest_ids(VIEW_DIR / "train.json.gz", "train")
    validation_ids = manifest_ids(VIEW_DIR / "validation.json.gz", "validation")
    if train_ids & validation_ids:
        raise ValueError("Gate train/validation manifests overlap")

    reset_random_seed(args.seed)
    detector_payload = torch.load(DETECTOR, map_location=args.device, weights_only=False)
    detector = LocalRobotDetector(**detector_payload.get("model_config", {})).to(args.device)
    detector.load_state_dict(detector_payload["model_state_dict"])
    detector.eval()
    navigation = load_actor(NAVIGATION, args.device)
    interaction = load_actor(INTERACTION, args.device)
    loader_args = argparse.Namespace(batch_size=args.batch_size, max_tracks=4, device=args.device)
    a1_train = build_dataset(list_shards(A1_ROUTE / "local_data/shards/train"), detector, navigation, interaction, loader_args, "train")
    student_train = build_dataset(list_shards(B_ROUTE / "local_data/student_shards/train"), detector, navigation, interaction, loader_args, "train")
    validation = build_dataset(list_shards(A1_ROUTE / "local_data/shards/validation"), detector, navigation, interaction, loader_args, "validation")
    verify_dataset_ids(a1_train, train_ids, "A1 train")
    verify_dataset_ids(student_train, train_ids, "student train")
    verify_dataset_ids(validation, validation_ids, "validation")
    train = concatenate_sources((("a1", a1_train), ("student", student_train)))

    train_raw = np.concatenate((train.base_features, train.actor_features), axis=1)
    validation_raw = np.concatenate((validation.base_features, validation.actor_features), axis=1)
    train_norm, validation_norm, feature_mean, feature_std = normalize_source_balanced(
        train_raw, validation_raw, train.scenarios
    )
    phase_train_x, phase_train_indices = make_temporal_windows(train_norm, train.sequence_indices, 8)
    phase_validation_x, phase_validation_indices = make_temporal_windows(validation_norm, validation.sequence_indices, 8)
    phase_train_y = train.labels["any"][phase_train_indices].astype(np.float32)
    phase_validation_y = validation.labels["any"][phase_validation_indices].astype(np.float32)
    phase_train_w = source_scenario_sample_weights(train.labels["any"], train.scenarios)[phase_train_indices]

    reward_train = load_reward_sources(DEFAULT_REWARD_TRAIN)
    reward_validation = load_reward_sources(DEFAULT_REWARD_VALIDATION)
    reward_train_scenes = set(reward_train["scenario_ids"].tolist())
    reward_validation_scenes = set(reward_validation["scenario_ids"].tolist())
    if reward_train_scenes & reward_validation_scenes:
        raise ValueError("reward train/validation scenes overlap")
    if not reward_train_scenes <= train_ids or not reward_validation_scenes <= validation_ids:
        raise ValueError("reward records do not belong to their frozen manifests")
    reward_train_x = normalize_padded_windows(reward_train["gate_windows"], reward_train["history_lengths"], feature_mean, feature_std)
    reward_validation_x = normalize_padded_windows(reward_validation["gate_windows"], reward_validation["history_lengths"], feature_mean, feature_std)
    scale, unique_train_anchors = robust_scale_by_anchor(reward_train["anchor_ids"], reward_train["mean_delta_r"])
    reward_train_y = bounded_reward_target(torch.from_numpy(reward_train["mean_delta_r"]), scale).numpy().astype(np.float32)
    reward_validation_y = bounded_reward_target(torch.from_numpy(reward_validation["mean_delta_r"]), scale).numpy().astype(np.float32)
    reward_train_w = scenario_weights(reward_train["scenario_ids"])
    reward_validation_w = scenario_weights(reward_validation["scenario_ids"])

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = DualEncoderRewardTemporalGate(82, 64).to(args.device)
    phase_epoch, phase_history = train_phase(
        model, phase_train_x, phase_train_y, phase_train_w,
        phase_validation_x, phase_validation_y, args
    )
    reward_epoch, reward_validation_metrics, reward_history = train_reward(
        model, reward_train_x, reward_train_y, reward_train_w,
        reward_validation_x, reward_validation_y, reward_validation_w,
        reward_validation["mean_delta_r"], args
    )
    base_phase, joint_phase = phase_metrics(
        model, phase_validation_x, phase_validation_indices, validation,
        args.batch_size, args.device, 0.25
    )
    constant = float(np.median(reward_train_y))
    constant_huber = weighted_huber(
        np.full_like(reward_validation_y, constant), reward_validation_y, reward_validation_w
    )
    criteria = {
        "reward_beats_train_median_constant": reward_validation_metrics["weighted_huber"] < constant_huber,
        "reward_pearson_positive": reward_validation_metrics["pearson"] is not None and reward_validation_metrics["pearson"] > 0.0,
        "joint_f1_drop_at_most_0_02": joint_phase["f1"] >= base_phase["f1"] - 0.02,
        "joint_fpr_increase_at_most_0_02": joint_phase["fpr"] <= base_phase["fpr"] + 0.02,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = args.output_dir / "best.pt"
    decision = {
        "score": "sigmoid(interaction_logit + reward_weight * bounded_advantage)",
        "reward_weight": 0.25,
        "switch_on_threshold": 0.43,
        "switch_off_threshold": 0.33,
        "minimum_hold_router_updates": 3,
        "evaluation_stride_environment_steps": 2,
    }
    checkpoint = {
        "format_version": 1,
        "model_id": "G34-dual-encoder-reward-aware",
        "label": "joint-2m-phase-and-counterfactual-reward",
        "model_config": {"input_dim": 82, "hidden_dim": 64},
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "feature_set": "base_and_actor_actions",
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "threshold": 0.43,
        "sequence_length": 8,
        "max_tracks": 4,
        "reward_target_scale": scale,
        "router_decision": decision,
    }
    torch.save(checkpoint, checkpoint_path)
    summary = {
        "format_version": 1,
        "protocol": "G34-dual-encoder-joint-supervision-v1",
        "seed": args.seed,
        "inputs": {str(path.relative_to(ROOT)): sha256_file(path) for path in all_inputs},
        "samples": {
            "phase_train_frames": len(phase_train_x),
            "phase_validation_frames": len(phase_validation_x),
            "reward_train_records": len(reward_train_y),
            "reward_train_unique_anchors": unique_train_anchors,
            "reward_train_unique_scenes": len(reward_train_scenes),
            "reward_validation_records": len(reward_validation_y),
            "reward_validation_unique_scenes": len(reward_validation_scenes),
        },
        "reward_target": {
            "definition": "tanh(clip(delta_r,-scale,scale)/scale)",
            "scale_p95_abs_unique_anchor_median_train_only": scale,
            "terminal_samples_retained": True,
        },
        "best_phase_epoch": phase_epoch,
        "best_reward_epoch": reward_epoch,
        "reward_validation": reward_validation_metrics,
        "reward_constant_baseline_huber": constant_huber,
        "phase_validation": {"base": base_phase, "joint": joint_phase},
        "offline_criteria": criteria,
        "offline_admission_passed": all(criteria.values()),
        "phase_history": phase_history,
        "reward_history": reward_history,
        "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "samples": summary["samples"],
        "reward_validation": reward_validation_metrics,
        "reward_constant_baseline_huber": constant_huber,
        "phase_base_f1_fpr": [base_phase["f1"], base_phase["fpr"]],
        "phase_joint_f1_fpr": [joint_phase["f1"], joint_phase["fpr"]],
        "offline_criteria": criteria,
        "offline_admission_passed": summary["offline_admission_passed"],
    }, indent=2))
    if not summary["offline_admission_passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
