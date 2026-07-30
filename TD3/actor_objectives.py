import torch
import torch.nn.functional as F


def approaching_safety_mask(
    critic_states,
    actor_state_dim,
    context_feature_dim,
    safety_distance,
    min_closing_speed,
):
    if context_feature_dim < 7:
        raise ValueError("Approaching-neighbor loss requires ego-motion context")
    contexts = critic_states[:, actor_state_dim:]
    if contexts.shape[1] % context_feature_dim != 0:
        raise ValueError("Critic context does not match context_feature_dim")

    slots = contexts.reshape(len(contexts), -1, context_feature_dim)
    position = slots[:, :, :2]
    distance = slots[:, :, 2]
    relative_velocity = slots[:, :, 4:6]
    valid = slots[:, :, context_feature_dim - 1] > 0.5
    radial_velocity = torch.sum(position * relative_velocity, dim=2) / torch.clamp(
        distance, min=1e-6
    )
    closing_speed = -radial_velocity
    approaching = (
        valid
        & (distance <= safety_distance)
        & (closing_speed >= min_closing_speed)
    )
    return torch.any(approaching, dim=1)


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


def actor_slowdown_safety_loss(
    actor_actions,
    critic_states,
    actor_state_dim,
    context_feature_dim,
    safety_distance,
    min_closing_speed,
    max_safe_linear_action,
):
    """Penalize the current Actor for moving too fast in close approaching states."""
    safety_mask = approaching_safety_mask(
        critic_states,
        actor_state_dim,
        context_feature_dim,
        safety_distance,
        min_closing_speed,
    )
    selected_count = int(torch.sum(safety_mask).item())
    if selected_count == 0:
        return actor_actions.sum() * 0.0, 0

    linear_action = actor_actions[safety_mask, 0]
    excess_speed = F.relu(linear_action - float(max_safe_linear_action))
    return torch.mean(excess_speed * excess_speed), selected_count
