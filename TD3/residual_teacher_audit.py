import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorScaledResidual(nn.Module):
    """Small audit-only adapter with independently calibrated action ranges."""

    def __init__(self, state_dim, action_dim, hidden_dim, residual_scale):
        super().__init__()
        scale = torch.as_tensor(residual_scale, dtype=torch.float32)
        if scale.shape != (action_dim,) or torch.any(scale <= 0.0):
            raise ValueError("residual_scale must contain one positive value per action")
        self.layer_1 = nn.Linear(state_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, action_dim)
        self.register_buffer("residual_scale", scale)

    def forward(self, state):
        hidden = F.relu(self.layer_1(state))
        return self.residual_scale * torch.tanh(self.layer_2(hidden))


def interaction_labels_from_critic_states(
    critic_states,
    state_dim,
    context_feature_dim,
    distance_threshold,
):
    values = np.asarray(critic_states, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] <= state_dim:
        raise ValueError("critic_states must contain Actor state and neighbor context")
    context_width = values.shape[1] - state_dim
    if context_feature_dim < 4 or context_width % context_feature_dim:
        raise ValueError("neighbor context width is incompatible with feature dimension")
    if distance_threshold <= 0.0:
        raise ValueError("distance_threshold must be positive")
    slots = values[:, state_dim:].reshape(
        len(values), -1, context_feature_dim
    )
    valid = slots[:, :, context_feature_dim - 1] > 0.5
    distances = np.where(valid, slots[:, :, 2], np.inf)
    return np.min(distances, axis=1) <= float(distance_threshold)


def calibrate_residual_scale(
    interaction_deltas,
    quantile=0.99,
    minimum=0.05,
    maximum=2.0,
):
    values = np.asarray(interaction_deltas, dtype=np.float32)
    if values.ndim != 2 or not len(values):
        raise ValueError("interaction_deltas must be a non-empty matrix")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in (0, 1]")
    if minimum <= 0.0 or maximum < minimum:
        raise ValueError("invalid residual scale limits")
    scale = np.quantile(np.abs(values), quantile, axis=0)
    return np.clip(scale, minimum, maximum).astype(np.float32)


def balanced_class_weights(labels):
    labels = np.asarray(labels, dtype=bool)
    positive = int(np.sum(labels))
    negative = int(len(labels) - positive)
    if positive == 0 or negative == 0:
        raise ValueError("both normal and interaction states are required")
    weights = np.where(labels, 0.5 / positive, 0.5 / negative)
    return (weights * len(labels)).astype(np.float32)


def action_error_metrics(predicted, target):
    predicted = np.asarray(predicted, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError("predicted and target actions must be aligned matrices")
    error = predicted - target
    return {
        "mse": float(np.mean(np.square(error))),
        "mae": float(np.mean(np.abs(error))),
        "per_action_mae": np.mean(np.abs(error), axis=0).astype(float).tolist(),
        "mean": np.mean(predicted, axis=0).astype(float).tolist(),
    }


def teacher_choice_accuracy(
    residual,
    generalist_actions,
    specialist_actions,
    labels,
    disagreement_threshold=0.05,
):
    residual = np.asarray(residual, dtype=np.float32)
    generalist = np.asarray(generalist_actions, dtype=np.float32)
    specialist = np.asarray(specialist_actions, dtype=np.float32)
    labels = np.asarray(labels, dtype=bool)
    if residual.shape != generalist.shape or generalist.shape != specialist.shape:
        raise ValueError("actions and residual must have identical shapes")
    if labels.shape != generalist.shape[:1]:
        raise ValueError("labels must align with actions")
    student = np.clip(generalist + residual, -1.0, 1.0)
    selected = np.where(labels[:, None], specialist, generalist)
    other = np.where(labels[:, None], generalist, specialist)
    selected_distance = np.mean(np.square(student - selected), axis=1)
    other_distance = np.mean(np.square(student - other), axis=1)
    informative = (
        np.max(np.abs(generalist - specialist), axis=1)
        > float(disagreement_threshold)
    )
    result = {}
    for name, mask in (
        ("all", informative),
        ("normal", informative & ~labels),
        ("interaction", informative & labels),
    ):
        result[name] = {
            "frames": int(np.sum(mask)),
            "accuracy": (
                float(np.mean(selected_distance[mask] < other_distance[mask]))
                if np.any(mask)
                else None
            ),
        }
    return result
