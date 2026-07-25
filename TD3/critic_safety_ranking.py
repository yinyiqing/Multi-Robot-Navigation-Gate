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
        raise ValueError("Approaching-neighbor ranking requires ego-motion context")
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


def critic_safety_ranking_loss(
    critic,
    critic_states,
    actions,
    actor_state_dim,
    context_feature_dim,
    safety_distance,
    min_closing_speed,
    linear_action_delta,
    margin,
):
    safety_mask = approaching_safety_mask(
        critic_states,
        actor_state_dim,
        context_feature_dim,
        safety_distance,
        min_closing_speed,
    )
    slower_actions = actions.clone()
    slower_actions[:, 0] = torch.clamp(
        slower_actions[:, 0] - linear_action_delta, min=-1.0, max=1.0
    )
    changed = torch.abs(slower_actions[:, 0] - actions[:, 0]) > 1e-6
    selected = safety_mask & changed
    selected_count = int(torch.sum(selected).item())
    if selected_count == 0:
        return next(critic.parameters()).sum() * 0.0, 0

    fast_q1, fast_q2 = critic(critic_states[selected], actions[selected])
    slow_q1, slow_q2 = critic(critic_states[selected], slower_actions[selected])
    loss = F.relu(fast_q1 - slow_q1 + margin).mean()
    loss = loss + F.relu(fast_q2 - slow_q2 + margin).mean()
    return loss, selected_count
