#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "TD3"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from actor_models import Actor
from interaction_gate import InteractionGate
from robot_perception.dataset import list_shards, load_shard
from robot_perception.gate_features import build_gate_feature, gate_feature_dim
from robot_perception.models import LocalRobotDetector
from robot_perception.tracker import RobotCandidateTracker
from temporal_interaction_gate import (
    ACTOR_COMPARISON_DIM,
    TemporalInteractionGate,
    actor_comparison_features,
)
from train_interaction_gate import (
    GATE_KEYS,
    binary_metrics,
    detector_probabilities,
    gate_metrics,
    sample_weights,
    select_threshold,
)


EXPECTED_STANDARD_ACTOR_SHA256 = (
    "fa28855049b67b3ee44c66d55d4f14441fc7c521e5429862c75b152f7d5cacc5"
)
EXPECTED_INTERACTION_ACTOR_SHA256 = (
    "6ec1942fcd497ab1cc2a85a5aaec8f524395dc21ff21a442dca243a52e917c0b"
)
EXPECTED_DETECTOR_SHA256 = (
    "0b914c0d090bbaba0a2be63c0d75d88580bf4d71778a7f1749c63989e3dbbd56"
)
EXPECTED_A1_MANIFEST_SHA256 = {
    "train": "a97534ed22d1b4b12951cd42b80d515037774830f38cb432926c0e9f50379026",
    "validation": "e261a7afbac8f7341ab13609c2662a2824a0ff383789287ad7733290389cd99d",
}
LABEL_KEYS = {
    "front": "frame_front_interaction_labels",
    "any": "frame_oracle_interaction_labels",
}


@dataclass
class GateDataset:
    base_features: np.ndarray
    actor_features: np.ndarray
    labels: dict
    scenarios: np.ndarray
    strata: np.ndarray
    sequence_indices: list


def parse_args():
    route = (
        PROJECT_ROOT
        / "experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究"
        / "G11_A_已有数据时序pilot"
    )
    old_gate = (
        PROJECT_ROOT
        / "experiments/03_保留专门化/02_论文主线/results/06_Gate开发"
        / "D5_G2_interaction_gate_v1"
    )
    parser = argparse.ArgumentParser(
        description="Run the G11-A0 offline temporal Gate diagnostic."
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=old_gate / "local_data/shards/pilot_train",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=old_gate / "local_data/shards/pilot_validation",
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
        "--reference-front-checkpoint",
        type=Path,
        default=old_gate / "local_data/model/oracle_front_v1/best.pt",
    )
    parser.add_argument(
        "--reference-any-checkpoint",
        type=Path,
        default=old_gate / "local_data/model/oracle_any_v1/best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=route / "local_data/seed20260727",
    )
    parser.add_argument(
        "--experiment-id", choices=["G11-A0", "G11-A1"], default="G11-A0"
    )
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument(
        "--threshold-policy",
        choices=["legacy", "match-s0-fpr"],
        default="legacy",
    )
    parser.add_argument("--labels", nargs="+", choices=sorted(LABEL_KEYS), default=["front", "any"])
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["S0", "S1", "T1"],
        default=["S0", "S1", "T1"],
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--temporal-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--temporal-hidden-dim", type=int, default=64)
    parser.add_argument("--max-tracks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_relative_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def verify_frozen_actor(path, expected_sha256):
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            "frozen Actor hash mismatch for %s: expected %s, got %s"
            % (path, expected_sha256, actual)
        )
    return actual


def reset_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_actor(path, device):
    model = Actor(24, 2).to(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


@torch.no_grad()
def actor_features_for_states(standard_actor, interaction_actor, states, device):
    values = torch.from_numpy(np.asarray(states, dtype=np.float32)).to(device)
    standard = standard_actor(values).cpu().numpy()
    interaction = interaction_actor(values).cpu().numpy()
    return actor_comparison_features(standard, interaction)


def build_dataset(
    paths,
    detector,
    standard_actor,
    interaction_actor,
    args,
    expected_split=None,
):
    base_features = []
    action_features = []
    labels = {name: [] for name in LABEL_KEYS}
    scenarios = []
    strata = []
    sequence_indices = []
    global_offset = 0
    seen_scenarios = set()

    for path in paths:
        shard = load_shard(path)
        missing = GATE_KEYS - set(shard)
        if missing:
            raise ValueError("gate shard %s is missing %s" % (path, sorted(missing)))
        scenario_id = str(shard["scenario_id"])
        if scenario_id in seen_scenarios:
            raise ValueError("duplicate Gate shard scenario: %s" % scenario_id)
        seen_scenarios.add(scenario_id)
        if expected_split is not None and str(shard["split"]) != expected_split:
            raise ValueError(
                "Gate shard %s has split %s, expected %s"
                % (path, str(shard["split"]), expected_split)
            )
        if expected_split is not None and str(shard["interaction_band"]) not in (
            "weak",
            "interaction",
        ):
            raise ValueError("Gate shard %s has invalid interaction band" % path)
        probabilities = detector_probabilities(
            detector, shard["patches"], args.batch_size, args.device
        )
        frame_candidates = [[] for _ in range(len(shard["frame_ego_indices"]))]
        for candidate_index, frame_index in enumerate(shard["frame_record_indices"]):
            frame_candidates[int(frame_index)].append(candidate_index)

        shard_features = np.zeros(
            (
                len(frame_candidates),
                gate_feature_dim(24, args.max_tracks),
            ),
            dtype=np.float32,
        )
        for ego_index in np.unique(shard["frame_ego_indices"]):
            tracker = RobotCandidateTracker()
            frame_rows = np.flatnonzero(shard["frame_ego_indices"] == ego_index)
            frame_rows = frame_rows[
                np.argsort(shard["frame_indices_unique"][frame_rows], kind="stable")
            ]
            sequence_indices.append(global_offset + frame_rows.astype(np.int64))
            for frame_row in frame_rows:
                candidate_rows = np.asarray(
                    frame_candidates[int(frame_row)], dtype=np.int64
                )
                tracked = tracker.update(
                    shard["candidate_centers"][candidate_rows],
                    probabilities[candidate_rows],
                    shard["frame_ego_poses"][frame_row],
                    shard["frame_timestamps"][frame_row],
                )
                shard_features[frame_row] = build_gate_feature(
                    shard["frame_actor_states"][frame_row],
                    tracked,
                    max_tracks=args.max_tracks,
                )

        count = len(shard_features)
        base_features.append(shard_features)
        action_features.append(
            actor_features_for_states(
                standard_actor,
                interaction_actor,
                shard["frame_actor_states"],
                args.device,
            )
        )
        for label_name, key in LABEL_KEYS.items():
            labels[label_name].append(shard[key].astype(np.float32))
        scenarios.extend([scenario_id] * count)
        strata.extend(
            [f"{str(shard['scenario_pool'])}_{str(shard['interaction_band'])}"]
            * count
        )
        global_offset += count

    result = GateDataset(
        base_features=np.concatenate(base_features),
        actor_features=np.concatenate(action_features),
        labels={name: np.concatenate(values) for name, values in labels.items()},
        scenarios=np.asarray(scenarios),
        strata=np.asarray(strata),
        sequence_indices=sequence_indices,
    )
    if len(result.base_features) != global_offset:
        raise AssertionError("frame accounting mismatch")
    covered = np.concatenate(result.sequence_indices)
    if len(covered) != global_offset or set(covered.tolist()) != set(range(global_offset)):
        raise ValueError("each frame must belong to exactly one ego sequence")
    return result


def normalize_features(train_features, validation_features):
    mean = np.mean(train_features, axis=0).astype(np.float32)
    std = np.std(train_features, axis=0).astype(np.float32)
    std[std < 1e-3] = 1.0
    return (
        ((train_features - mean) / std).astype(np.float32),
        ((validation_features - mean) / std).astype(np.float32),
        mean,
        std,
    )


def make_temporal_windows(features, sequence_indices, sequence_length):
    if features.ndim != 2 or sequence_length < 2:
        raise ValueError("temporal windows require [N, D] features and length >= 2")
    windows = []
    target_indices = []
    for indices in sequence_indices:
        sequence = features[indices]
        for position, target_index in enumerate(indices):
            start = max(0, position - sequence_length + 1)
            history = sequence[start : position + 1]
            window = np.zeros((sequence_length, features.shape[1]), dtype=np.float32)
            window[-len(history) :] = history
            windows.append(window)
            target_indices.append(int(target_index))
    return np.stack(windows), np.asarray(target_indices, dtype=np.int64)


@torch.no_grad()
def predict(model, features, batch_size, device):
    model.eval()
    probabilities = []
    for start in range(0, len(features), batch_size):
        values = torch.from_numpy(features[start : start + batch_size]).to(device)
        probabilities.append(torch.sigmoid(model(values)).cpu().numpy())
    return np.concatenate(probabilities)


def average_precision(probabilities, labels):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    positive_count = int(np.sum(labels))
    if positive_count == 0:
        return None
    order = np.argsort(-probabilities, kind="stable")
    ordered = labels[order]
    cumulative = np.cumsum(ordered)
    precision = cumulative / np.arange(1, len(ordered) + 1)
    return float(np.sum(precision[ordered]) / positive_count)


def _positive_intervals(values):
    values = np.asarray(values, dtype=bool)
    intervals = []
    start = None
    for index, value in enumerate(values):
        if value and start is None:
            start = index
        elif not value and start is not None:
            intervals.append((start, index))
            start = None
    if start is not None:
        intervals.append((start, len(values)))
    return intervals


def event_metrics(probabilities, labels, sequence_indices, threshold):
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels, dtype=bool)
    predictions = probabilities >= threshold
    true_events = 0
    predicted_events = 0
    matched_true = 0
    matched_predicted = 0
    onset_delays = []
    intersection = 0
    union = 0
    switches = []

    for indices in sequence_indices:
        truth = labels[indices]
        prediction = predictions[indices]
        true_intervals = _positive_intervals(truth)
        predicted_intervals = _positive_intervals(prediction)
        true_events += len(true_intervals)
        predicted_events += len(predicted_intervals)
        for true_start, true_end in true_intervals:
            overlaps = [
                item
                for item in predicted_intervals
                if item[0] < true_end and true_start < item[1]
            ]
            if overlaps:
                matched_true += 1
                onset_delays.append(overlaps[0][0] - true_start)
        for predicted_start, predicted_end in predicted_intervals:
            if any(
                true_start < predicted_end and predicted_start < true_end
                for true_start, true_end in true_intervals
            ):
                matched_predicted += 1
        intersection += int(np.sum(truth & prediction))
        union += int(np.sum(truth | prediction))
        switches.append(
            int(prediction[0])
            + int(np.sum(prediction[1:] != prediction[:-1]))
            if len(prediction)
            else 0
        )

    return {
        "true_events": true_events,
        "predicted_events": predicted_events,
        "event_recall": matched_true / true_events if true_events else None,
        "event_precision": (
            matched_predicted / predicted_events if predicted_events else None
        ),
        "positive_interval_iou": intersection / union if union else None,
        "mean_onset_delay_steps": (
            float(np.mean(onset_delays)) if onset_delays else None
        ),
        "total_switches": int(np.sum(switches)),
        "mean_switches_per_ego_sequence": float(np.mean(switches)),
    }


def evaluate_probabilities(probabilities, dataset, label_name, threshold):
    labels = dataset.labels[label_name]
    guard_mask = dataset.strata == "standard_weak"
    metrics = gate_metrics(probabilities, labels, threshold, guard_mask)
    weak_mask = np.asarray(
        [str(value).endswith("_weak") for value in dataset.strata], dtype=bool
    )
    if np.any(weak_mask):
        metrics["weak_fpr"] = binary_metrics(
            probabilities[weak_mask], labels[weak_mask], threshold
        )["fpr"]
    metrics["average_precision"] = average_precision(probabilities, labels)
    metrics["events"] = event_metrics(
        probabilities,
        labels,
        dataset.sequence_indices,
        threshold,
    )
    return metrics


def select_threshold_with_fpr_caps(
    probabilities, labels, weak_mask, fpr_caps=None
):
    candidates = []
    for threshold in np.linspace(0.01, 0.99, 99):
        metrics = gate_metrics(probabilities, labels, threshold, weak_mask)
        metrics["weak_fpr"] = metrics.pop("standard_weak_fpr")
        meets_caps = fpr_caps is None or (
            metrics["fpr"] <= fpr_caps["fpr"] + 1e-12
            and metrics["weak_fpr"] <= fpr_caps["weak_fpr"] + 1e-12
        )
        metrics["meets_fpr_caps"] = bool(meets_caps)
        candidates.append(metrics)
    feasible = [item for item in candidates if item["meets_fpr_caps"]]
    if not feasible:
        raise ValueError("no threshold satisfies the frozen S0 FPR caps")
    return max(feasible, key=lambda item: (item["f1"], item["recall"], -item["fpr"]))


def evaluate_reference(path, dataset, label_name, args):
    payload = torch.load(path, map_location=args.device, weights_only=False)
    if payload["label"] != label_name:
        raise ValueError("reference checkpoint label mismatch: %s" % path)
    model = InteractionGate(**payload["model_config"]).to(args.device)
    model.load_state_dict(payload["model_state_dict"])
    normalized = (
        dataset.base_features - np.asarray(payload["feature_mean"])
    ) / np.asarray(payload["feature_std"])
    probabilities = predict(model, normalized.astype(np.float32), args.batch_size, args.device)
    return {
        "checkpoint": project_relative_path(path),
        "sha256": sha256_file(path),
        "metrics": evaluate_probabilities(
            probabilities, dataset, label_name, float(payload["threshold"])
        ),
    }


def train_candidate(
    model_id,
    model,
    train_features,
    validation_features,
    train_dataset,
    validation_dataset,
    train_indices,
    validation_indices,
    label_name,
    batch_size,
    mean,
    std,
    args,
    fpr_caps=None,
):
    reset_random_seed(args.seed)
    model.apply(lambda module: module.reset_parameters() if hasattr(module, "reset_parameters") else None)
    model = model.to(args.device)
    train_labels = train_dataset.labels[label_name][train_indices]
    validation_labels = validation_dataset.labels[label_name][validation_indices]
    train_scenarios = train_dataset.scenarios[train_indices]
    validation_strata = validation_dataset.strata[validation_indices]
    guard_mask = validation_strata == "standard_weak"
    weak_mask = np.asarray(
        [str(value).endswith("_weak") for value in validation_strata], dtype=bool
    )
    weights = sample_weights(train_labels, train_scenarios)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    rng = np.random.default_rng(args.seed)
    best = None
    history = []

    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(train_features))
        model.train()
        losses = []
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            features = torch.from_numpy(train_features[batch]).to(args.device)
            labels = torch.from_numpy(train_labels[batch]).to(args.device)
            batch_weights = torch.from_numpy(weights[batch]).to(args.device)
            loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    model(features), labels, reduction="none"
                )
                * batch_weights
            ).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        probabilities = predict(
            model, validation_features, batch_size, args.device
        )
        if args.threshold_policy == "match-s0-fpr":
            metrics = select_threshold_with_fpr_caps(
                probabilities, validation_labels, weak_mask, fpr_caps
            )
        else:
            metrics = select_threshold(
                probabilities, validation_labels, guard_mask
            )
        metrics["average_precision"] = average_precision(
            probabilities, validation_labels
        )
        record = {"epoch": epoch, "loss": float(np.mean(losses)), **metrics}
        history.append(record)
        if args.threshold_policy == "match-s0-fpr":
            key = (metrics["f1"], metrics["recall"], -metrics["fpr"])
        else:
            key = (
                int(metrics["meets_entry_criteria"]),
                metrics["f1"],
                metrics["recall"],
                -metrics["fpr"],
            )
        if best is None or key > best[0]:
            best = (
                key,
                record,
                {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                },
            )
        print(
            "%s/%s epoch=%02d loss=%.5f f1=%.3f recall=%.3f fpr=%.3f pass=%s"
            % (
                label_name,
                model_id,
                epoch,
                record["loss"],
                metrics["f1"],
                metrics["recall"],
                metrics["fpr"],
                metrics["meets_entry_criteria"],
            )
        )

    model.load_state_dict(best[2])
    validation_probabilities_ordered = predict(
        model, validation_features, batch_size, args.device
    )
    validation_probabilities = np.zeros(
        len(validation_dataset.base_features), dtype=np.float32
    )
    validation_probabilities[validation_indices] = validation_probabilities_ordered
    train_probabilities_ordered = predict(model, train_features, batch_size, args.device)
    train_probabilities = np.zeros(len(train_dataset.base_features), dtype=np.float32)
    train_probabilities[train_indices] = train_probabilities_ordered
    threshold = float(best[1]["threshold"])

    output_dir = args.output_dir / label_name / model_id
    output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(model, TemporalInteractionGate):
        model_config = {
            "input_dim": model.input_dim,
            "hidden_dim": model.hidden_dim,
        }
    else:
        model_config = {
            "input_dim": model.input_dim,
            "hidden_dims": model.hidden_dims,
        }
    checkpoint = {
        "format_version": 1,
        "diagnostic_only": args.experiment_id == "G11-A0",
        "model_id": model_id,
        "label": label_name,
        "model_state_dict": best[2],
        "model_config": model_config,
        "feature_set": "base" if model_id == "S0" else "base_and_actor_actions",
        "feature_mean": mean,
        "feature_std": std,
        "threshold": threshold,
        "sequence_length": args.sequence_length if model_id == "T1" else 1,
        "experiment_id": args.experiment_id,
        "threshold_policy": args.threshold_policy,
        "fpr_caps": fpr_caps,
    }
    torch.save(checkpoint, output_dir / "best.pt")
    return {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": int(best[1]["epoch"]),
        "threshold": threshold,
        "train": evaluate_probabilities(
            train_probabilities, train_dataset, label_name, threshold
        ),
        "validation": evaluate_probabilities(
            validation_probabilities, validation_dataset, label_name, threshold
        ),
        "history": history,
        "checkpoint": project_relative_path(output_dir / "best.pt"),
    }


def main():
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.device != "cpu" and not torch.cuda.is_available():
        raise ValueError("requested device is unavailable: %s" % args.device)
    if args.epochs < 1 or args.sequence_length < 2 or args.max_tracks < 1:
        raise ValueError("training dimensions must be positive")
    if args.experiment_id == "G11-A1":
        if args.labels != ["any"] or set(args.models) != {"S0", "T1"}:
            raise ValueError("G11-A1 requires --labels any --models S0 T1")
        if args.threshold_policy != "match-s0-fpr":
            raise ValueError("G11-A1 requires --threshold-policy match-s0-fpr")
        if args.train_manifest is None or args.validation_manifest is None:
            raise ValueError("G11-A1 requires frozen train and validation manifests")
    elif args.threshold_policy != "legacy":
        raise ValueError("G11-A0 must preserve the legacy threshold policy")
    standard_hash = verify_frozen_actor(
        args.standard_actor, EXPECTED_STANDARD_ACTOR_SHA256
    )
    interaction_hash = verify_frozen_actor(
        args.interaction_actor, EXPECTED_INTERACTION_ACTOR_SHA256
    )
    detector_hash = verify_frozen_actor(
        args.detector_checkpoint, EXPECTED_DETECTOR_SHA256
    )
    reset_random_seed(args.seed)
    detector_payload = torch.load(
        args.detector_checkpoint, map_location=args.device, weights_only=False
    )
    detector = LocalRobotDetector(
        **detector_payload.get("model_config", {})
    ).to(args.device)
    detector.load_state_dict(detector_payload["model_state_dict"])
    detector.eval()
    standard_actor = load_actor(args.standard_actor, args.device)
    interaction_actor = load_actor(args.interaction_actor, args.device)

    train_dataset = build_dataset(
        list_shards(args.train_dir),
        detector,
        standard_actor,
        interaction_actor,
        args,
        "train" if args.experiment_id == "G11-A1" else None,
    )
    validation_dataset = build_dataset(
        list_shards(args.validation_dir),
        detector,
        standard_actor,
        interaction_actor,
        args,
        "validation" if args.experiment_id == "G11-A1" else None,
    )
    manifest_records = {}
    if args.experiment_id == "G11-A1":
        manifest_ids = {}
        for split, path in (
            ("train", args.train_manifest),
            ("validation", args.validation_manifest),
        ):
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            manifest_hash = sha256_file(path)
            if manifest_hash != EXPECTED_A1_MANIFEST_SHA256[split]:
                raise ValueError("G11-A1 %s manifest hash mismatch" % split)
            ids = {item["scenario_id"] for item in payload["scenarios"]}
            dataset = train_dataset if split == "train" else validation_dataset
            observed = set(dataset.scenarios.tolist())
            if observed != ids:
                raise ValueError(
                    "%s shard/manifest mismatch: missing=%s extra=%s"
                    % (split, sorted(ids - observed)[:5], sorted(observed - ids)[:5])
                )
            manifest_ids[split] = ids
            manifest_records[split] = {
                "path": project_relative_path(path),
                "sha256": manifest_hash,
                "scenarios": len(ids),
            }
        if manifest_ids["train"] & manifest_ids["validation"]:
            raise ValueError("G11-A1 train/validation manifests overlap")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_train, base_validation, base_mean, base_std = normalize_features(
        train_dataset.base_features, validation_dataset.base_features
    )
    augmented_train_raw = np.concatenate(
        (train_dataset.base_features, train_dataset.actor_features), axis=1
    )
    augmented_validation_raw = np.concatenate(
        (validation_dataset.base_features, validation_dataset.actor_features), axis=1
    )
    augmented_train, augmented_validation, augmented_mean, augmented_std = (
        normalize_features(augmented_train_raw, augmented_validation_raw)
    )
    temporal_train, temporal_train_indices = make_temporal_windows(
        augmented_train, train_dataset.sequence_indices, args.sequence_length
    )
    temporal_validation, temporal_validation_indices = make_temporal_windows(
        augmented_validation,
        validation_dataset.sequence_indices,
        args.sequence_length,
    )
    static_train_indices = np.arange(len(train_dataset.base_features))
    static_validation_indices = np.arange(len(validation_dataset.base_features))

    references = {}
    reference_paths = {
        "front": args.reference_front_checkpoint,
        "any": args.reference_any_checkpoint,
    }
    results = {}
    for label_name in args.labels:
        references[label_name] = evaluate_reference(
            reference_paths[label_name], validation_dataset, label_name, args
        )
        results[label_name] = {}
        fpr_caps = None
        if "S0" in args.models:
            results[label_name]["S0"] = train_candidate(
                "S0",
                InteractionGate(base_train.shape[1]),
                base_train,
                base_validation,
                train_dataset,
                validation_dataset,
                static_train_indices,
                static_validation_indices,
                label_name,
                args.batch_size,
                base_mean,
                base_std,
                args,
            )
            if args.threshold_policy == "match-s0-fpr":
                baseline = results[label_name]["S0"]["validation"]
                fpr_caps = {
                    "fpr": baseline["fpr"],
                    "weak_fpr": baseline["weak_fpr"],
                }
        if "S1" in args.models:
            results[label_name]["S1"] = train_candidate(
                "S1",
                InteractionGate(augmented_train.shape[1]),
                augmented_train,
                augmented_validation,
                train_dataset,
                validation_dataset,
                static_train_indices,
                static_validation_indices,
                label_name,
                args.batch_size,
                augmented_mean,
                augmented_std,
                args,
                fpr_caps,
            )
        if "T1" in args.models:
            results[label_name]["T1"] = train_candidate(
                "T1",
                TemporalInteractionGate(
                    augmented_train.shape[1], args.temporal_hidden_dim
                ),
                temporal_train,
                temporal_validation,
                train_dataset,
                validation_dataset,
                temporal_train_indices,
                temporal_validation_indices,
                label_name,
                args.temporal_batch_size,
                augmented_mean,
                augmented_std,
                args,
                fpr_caps,
            )

    summary = {
        "protocol": {
            "experiment_id": args.experiment_id,
            "diagnostic_only": args.experiment_id == "G11-A0",
            "reason": (
                "legacy positive-edge shards include multi-edge scenarios"
                if args.experiment_id == "G11-A0"
                else "current-protocol offline Gate pilot"
            ),
            "sealed_test_read": False,
            "device": args.device,
            "seed": args.seed,
            "epochs": args.epochs,
            "sequence_length": args.sequence_length,
            "models": args.models,
            "threshold_policy": args.threshold_policy,
            "base_feature_dim": gate_feature_dim(24, args.max_tracks),
            "actor_comparison_dim": ACTOR_COMPARISON_DIM,
            "standard_actor_sha256": standard_hash,
            "interaction_actor_sha256": interaction_hash,
            "detector_sha256": detector_hash,
            "train_shards": len(list_shards(args.train_dir)),
            "validation_shards": len(list_shards(args.validation_dir)),
            "manifests": manifest_records,
        },
        "samples": {
            "train_frames": len(train_dataset.base_features),
            "validation_frames": len(validation_dataset.base_features),
            "train_ego_sequences": len(train_dataset.sequence_indices),
            "validation_ego_sequences": len(validation_dataset.sequence_indices),
            "train_positive_rate": {
                name: float(np.mean(train_dataset.labels[name])) for name in args.labels
            },
            "validation_positive_rate": {
                name: float(np.mean(validation_dataset.labels[name]))
                for name in args.labels
            },
        },
        "reference_checkpoints": references,
        "results": results,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(summary_path), "results": results}, indent=2))


if __name__ == "__main__":
    main()
