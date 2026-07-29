import torch
import torch.nn.functional as F


def safe_reference_mask(
    critic_states,
    actor_state_dim,
    context_feature_dim,
    safe_distance,
):
    if context_feature_dim < 3:
        raise ValueError("Context needs a distance and validity feature")
    contexts = critic_states[:, actor_state_dim:]
    if contexts.shape[1] % context_feature_dim != 0:
        raise ValueError("Critic context does not match context_feature_dim")

    slots = contexts.reshape(len(contexts), -1, context_feature_dim)
    valid = slots[:, :, context_feature_dim - 1] > 0.5
    distances = slots[:, :, 2]
    interaction = torch.any(valid & (distances <= safe_distance), dim=1)
    return ~interaction


def conservative_actor_objective(
    q_values,
    actor_actions,
    reference_actions=None,
    q_normalization_alpha=0.0,
    anchor_weight=0.0,
    anchor_mask=None,
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
        if anchor_mask is None:
            anchor_loss = F.mse_loss(actor_actions, reference_actions)
        else:
            anchor_mask = torch.as_tensor(
                anchor_mask, dtype=torch.bool, device=actor_actions.device
            ).reshape(-1)
            if anchor_mask.shape[0] != actor_actions.shape[0]:
                raise ValueError("anchor_mask must have one entry per actor action")
            if torch.any(anchor_mask):
                anchor_loss = F.mse_loss(
                    actor_actions[anchor_mask], reference_actions[anchor_mask]
                )

    return q_loss + float(anchor_weight) * anchor_loss, anchor_loss, q_scale
