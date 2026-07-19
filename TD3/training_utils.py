import math


def replay_done(truncated, terminated):
    """End the transition on either an environment terminal or a reset timeout."""
    return int(bool(truncated or terminated))


def episode_train_iterations(agent_samples, num_agents):
    """Keep updates proportional to collective environment steps, not inactive agents."""
    if num_agents < 1:
        raise ValueError("num_agents must be positive")
    return max(int(math.ceil(float(agent_samples) / num_agents)), 1)


def decay_exploration_noise(current, initial, minimum, decay_steps):
    """Linearly decay from the configured initial value to the minimum."""
    if decay_steps < 1:
        raise ValueError("decay_steps must be positive")
    if initial < minimum:
        raise ValueError("initial exploration noise must be >= minimum")
    decrement = (float(initial) - float(minimum)) / float(decay_steps)
    return max(float(minimum), float(current) - decrement)
