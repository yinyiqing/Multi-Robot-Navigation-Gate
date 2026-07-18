import math


def replay_done(truncated, terminated):
    """End the transition on either an environment terminal or a reset timeout."""
    return int(bool(truncated or terminated))


def episode_train_iterations(agent_samples, num_agents):
    """Keep updates proportional to collective environment steps, not inactive agents."""
    if num_agents < 1:
        raise ValueError("num_agents must be positive")
    return max(int(math.ceil(float(agent_samples) / num_agents)), 1)
