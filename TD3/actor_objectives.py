import torch
import torch.nn.functional as F


def conservative_actor_objective(
    q_values,
    actor_actions,
    reference_actions=None,
    q_normalization_alpha=0.0,
    anchor_weight=0.0,
):
    """Combine a scale-normalized Q objective with base-policy regularization."""
    if q_normalization_alpha < 0.0:
        raise ValueError("q_normalization_alpha must be non-negative")
    if anchor_weight < 0.0:
        raise ValueError("anchor_weight must be non-negative")

    q_scale = torch.ones((), dtype=q_values.dtype, device=q_values.device)
    if q_normalization_alpha > 0.0:
        q_scale = q_values.abs().mean().detach().clamp(min=1e-6)
        q_loss = -float(q_normalization_alpha) * q_values.mean() / q_scale
    else:
        q_loss = -q_values.mean()

    anchor_loss = torch.zeros((), dtype=q_values.dtype, device=q_values.device)
    if reference_actions is not None and anchor_weight > 0.0:
        anchor_loss = F.mse_loss(actor_actions, reference_actions)

    return q_loss + float(anchor_weight) * anchor_loss, anchor_loss, q_scale
