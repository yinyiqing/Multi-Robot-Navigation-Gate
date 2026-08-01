import math


def replay_done(truncated, terminated):
    """End the transition on either an environment terminal or a reset timeout."""
    return int(bool(truncated or terminated))


def apply_timeout_reward(rewards, terminated, truncated, timeout_reward):
    """Assign an explicit terminal reward to agents still active at truncation."""
    adjusted = list(rewards)
    if not truncated or timeout_reward is None:
        return adjusted
    if len(adjusted) != len(terminated):
        raise ValueError("rewards and terminated must have the same length")
    for idx, is_terminated in enumerate(terminated):
        if not is_terminated:
            adjusted[idx] = float(timeout_reward)
    return adjusted


def episode_train_iterations(agent_samples, num_agents):
    """Keep updates proportional to collective environment steps, not inactive agents."""
    if num_agents < 1:
        raise ValueError("num_agents must be positive")
    return max(int(math.ceil(float(agent_samples) / num_agents)), 1)


def replay_ready_for_updates(replay_size, minimum_size):
    """Delay all network updates until replay contains enough transitions."""
    if minimum_size < 0:
        raise ValueError("minimum replay size must be non-negative")
    return replay_size > 0 and replay_size >= minimum_size


def decay_exploration_noise(current, initial, minimum, decay_steps):
    """Linearly decay from the configured initial value to the minimum."""
    if decay_steps < 1:
        raise ValueError("decay_steps must be positive")
    if initial < minimum:
        raise ValueError("initial exploration noise must be >= minimum")
    decrement = (float(initial) - float(minimum)) / float(decay_steps)
    return max(float(minimum), float(current) - decrement)
