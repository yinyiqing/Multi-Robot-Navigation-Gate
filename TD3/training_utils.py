import numpy as np


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


def episode_train_iterations(environment_steps):
    """Run one gradient update per joint environment transition."""
    if environment_steps < 0:
        raise ValueError("environment_steps must be non-negative")
    return max(int(environment_steps), 1)


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


def exploratory_action(
    policy_action,
    noise_scale,
    max_action,
    randomize_linear=False,
    rng=None,
):
    """Apply TD3 exploration while optionally covering the raw linear action."""
    if noise_scale < 0.0:
        raise ValueError("noise_scale must be non-negative")
    if max_action <= 0.0:
        raise ValueError("max_action must be positive")
    action = np.asarray(policy_action, dtype=np.float32).copy()
    if action.ndim != 1 or action.size < 2:
        raise ValueError("policy_action must contain linear and angular actions")
    random_source = np.random if rng is None else rng
    if randomize_linear:
        action[0] = random_source.uniform(-max_action, max_action)
        action[1:] += random_source.normal(0.0, noise_scale, size=action.size - 1)
    else:
        action += random_source.normal(0.0, noise_scale, size=action.size)
    return np.clip(action, -max_action, max_action)
