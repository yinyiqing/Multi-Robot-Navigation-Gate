from collections import Counter
from pathlib import Path

import numpy as np
import torch

from robot_perception.dataset import load_shard
from robot_perception.gate_features import build_gate_feature, gate_feature_dim
from robot_perception.tracker import RobotCandidateTracker


CONSOLIDATION_KEYS = {
    "frame_actor_states",
    "frame_front_interaction_labels",
    "frame_oracle_interaction_labels",
    "interaction_band",
    "scenario_id",
    "scenario_pool",
}


def load_consolidation_dataset(paths, label="any", state_dim=24):
    if label not in ("any", "front"):
        raise ValueError("label must be 'any' or 'front'")
    label_key = (
        "frame_oracle_interaction_labels"
        if label == "any"
        else "frame_front_interaction_labels"
    )
    states = []
    labels = []
    scenarios = []
    strata = []
    for path in [Path(value) for value in paths]:
        with np.load(str(path), allow_pickle=False) as shard:
            missing = CONSOLIDATION_KEYS - set(shard.files)
            if missing:
                raise ValueError(
                    f"policy-consolidation shard {path} is missing {sorted(missing)}"
                )
            shard_states = np.asarray(
                shard["frame_actor_states"], dtype=np.float32
            )
            if shard_states.ndim != 2 or shard_states.shape[1] != state_dim:
                raise ValueError(
                    f"actor states in {path} must have shape (N, {state_dim})"
                )
            shard_labels = np.asarray(shard[label_key], dtype=np.float32)
            if shard_labels.shape != (len(shard_states),):
                raise ValueError(f"interaction labels in {path} do not match states")
            count = len(shard_states)
            states.append(shard_states)
            labels.append(shard_labels)
            scenarios.extend([str(shard["scenario_id"])] * count)
            strata.extend(
                [
                    f"{str(shard['scenario_pool'])}_{str(shard['interaction_band'])}"
                ]
                * count
            )
    if not states:
        raise ValueError("policy consolidation requires at least one shard")
    return {
        "states": np.concatenate(states),
        "features": np.concatenate(states),
        "labels": np.concatenate(labels),
        "scenarios": np.asarray(scenarios),
        "strata": np.asarray(strata),
    }


@torch.no_grad()
def _detector_probabilities(model, patches, batch_size, device):
    probabilities = []
    for start in range(0, len(patches), batch_size):
        values = torch.from_numpy(
            patches[start : start + batch_size].astype(np.float32)
        ).to(device)
        probabilities.append(torch.sigmoid(model(values)[0]).cpu().numpy())
    return np.concatenate(probabilities) if probabilities else np.empty(0)


def load_augmented_consolidation_dataset(
    paths,
    detector,
    batch_size,
    device,
    label="any",
    state_dim=24,
    max_tracks=4,
):
    dataset = load_consolidation_dataset(paths, label=label, state_dim=state_dim)
    features = []
    ordered_states = []
    for path in [Path(value) for value in paths]:
        shard = load_shard(path)
        probabilities = _detector_probabilities(
            detector, shard["patches"], batch_size, device
        )
        frame_candidates = [[] for _ in range(len(shard["frame_ego_indices"]))]
        for candidate_index, frame_index in enumerate(shard["frame_record_indices"]):
            frame_candidates[int(frame_index)].append(candidate_index)
        shard_features = np.zeros(
            (len(frame_candidates), gate_feature_dim(state_dim, max_tracks)),
            dtype=np.float32,
        )
        for ego_index in np.unique(shard["frame_ego_indices"]):
            tracker = RobotCandidateTracker()
            frame_rows = np.flatnonzero(shard["frame_ego_indices"] == ego_index)
            frame_rows = frame_rows[
                np.argsort(shard["frame_indices_unique"][frame_rows], kind="stable")
            ]
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
                    max_tracks=max_tracks,
                )
        features.append(shard_features)
        ordered_states.append(shard["frame_actor_states"].astype(np.float32))
    actor_states = np.concatenate(ordered_states)
    if not np.array_equal(actor_states, dataset["states"]):
        raise ValueError("augmented features do not preserve shard state ordering")
    dataset["features"] = np.concatenate(features)
    return dataset


def initialize_augmented_actor(student, source_actor, source_state_dim):
    source_state = source_actor.state_dict()
    student_state = student.state_dict()
    if student_state["layer_1.weight"].shape[1] < source_state_dim:
        raise ValueError("student input is smaller than the source actor input")
    if source_state["layer_1.weight"].shape[1] != source_state_dim:
        raise ValueError("source actor input does not match source_state_dim")
    student_state["layer_1.weight"].zero_()
    student_state["layer_1.weight"][:, :source_state_dim].copy_(
        source_state["layer_1.weight"]
    )
    for key in student_state:
        if key != "layer_1.weight":
            student_state[key].copy_(source_state[key])
    student.load_state_dict(student_state)


def scenario_class_weights(labels, scenarios):
    labels = np.asarray(labels, dtype=np.int64)
    scenarios = np.asarray(scenarios)
    if labels.ndim != 1 or scenarios.shape != labels.shape:
        raise ValueError("labels and scenarios must be aligned one-dimensional arrays")
    if not np.all(np.isin(labels, (0, 1))):
        raise ValueError("interaction labels must be binary")
    weights = np.empty(len(labels), dtype=np.float64)
    for label in (0, 1):
        selected = labels == label
        if not np.any(selected):
            raise ValueError("both normal and interaction frames are required")
        counts = Counter(scenarios[selected].tolist())
        weights[selected] = np.asarray(
            [1.0 / counts[value] for value in scenarios[selected]],
            dtype=np.float64,
        )
        weights[selected] *= 0.5 / float(np.sum(weights[selected]))
    return (weights / np.mean(weights)).astype(np.float32)


def select_teacher_actions(generalist_actions, specialist_actions, labels):
    generalist_actions = np.asarray(generalist_actions, dtype=np.float32)
    specialist_actions = np.asarray(specialist_actions, dtype=np.float32)
    labels = np.asarray(labels, dtype=bool)
    if generalist_actions.shape != specialist_actions.shape:
        raise ValueError("teacher action arrays must have identical shapes")
    if generalist_actions.ndim != 2 or labels.shape != generalist_actions.shape[:1]:
        raise ValueError("teacher actions and interaction labels are not aligned")
    return np.where(labels[:, None], specialist_actions, generalist_actions)


@torch.no_grad()
def actor_actions(actor, states, batch_size, device):
    states = np.asarray(states, dtype=np.float32)
    outputs = []
    for start in range(0, len(states), batch_size):
        values = torch.from_numpy(states[start : start + batch_size]).to(device)
        outputs.append(actor(values).cpu().numpy())
    if not outputs:
        return np.empty((0, 2), dtype=np.float32)
    return np.concatenate(outputs).astype(np.float32, copy=False)


def _action_error_metrics(student_actions, selected_actions):
    error = np.asarray(student_actions) - np.asarray(selected_actions)
    return {
        "mse": float(np.mean(np.square(error))),
        "mae": float(np.mean(np.abs(error))),
        "linear_mae": float(np.mean(np.abs(error[:, 0]))),
        "angular_mae": float(np.mean(np.abs(error[:, 1]))),
    }


def consolidation_metrics(
    student_actions,
    generalist_actions,
    specialist_actions,
    labels,
    strata=None,
    disagreement_threshold=0.05,
):
    student_actions = np.asarray(student_actions, dtype=np.float32)
    generalist_actions = np.asarray(generalist_actions, dtype=np.float32)
    specialist_actions = np.asarray(specialist_actions, dtype=np.float32)
    labels = np.asarray(labels, dtype=bool)
    selected = select_teacher_actions(
        generalist_actions, specialist_actions, labels
    )
    disagreement = np.max(
        np.abs(generalist_actions - specialist_actions), axis=1
    ) > float(disagreement_threshold)
    selected_distance = np.mean(np.square(student_actions - selected), axis=1)
    other = select_teacher_actions(specialist_actions, generalist_actions, labels)
    other_distance = np.mean(np.square(student_actions - other), axis=1)
    informative = disagreement & (np.abs(selected_distance - other_distance) > 1e-12)
    result = {
        "all": _action_error_metrics(student_actions, selected),
        "normal": _action_error_metrics(student_actions[~labels], selected[~labels]),
        "interaction": _action_error_metrics(student_actions[labels], selected[labels]),
        "frames": int(len(labels)),
        "interaction_frames": int(np.sum(labels)),
        "interaction_rate": float(np.mean(labels)),
        "teacher_disagreement_rate": float(np.mean(disagreement)),
        "teacher_choice_accuracy": (
            float(np.mean(selected_distance[informative] < other_distance[informative]))
            if np.any(informative)
            else None
        ),
        "teacher_choice_frames": int(np.sum(informative)),
    }
    if strata is not None:
        strata = np.asarray(strata)
        if strata.shape != labels.shape:
            raise ValueError("strata must align with labels")
        result["strata"] = {
            name: _action_error_metrics(student_actions[strata == name], selected[strata == name])
            for name in sorted(set(strata.tolist()))
        }
    return result
