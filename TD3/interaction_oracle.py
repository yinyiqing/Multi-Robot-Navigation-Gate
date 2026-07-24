import numpy as np


def nearest_context_distance(context, feature_dim=5):
    values = np.asarray(context, dtype=np.float32)
    if feature_dim < 4 or values.size % feature_dim:
        raise ValueError("Neighbor context is incompatible with feature_dim")
    slots = values.reshape(-1, feature_dim)
    valid = slots[:, feature_dim - 1] > 0.5
    if not np.any(valid):
        return float("inf")
    return float(np.min(slots[valid, 2]))


def interaction_mask(contexts, distance_threshold, feature_dim=5):
    if distance_threshold <= 0.0:
        raise ValueError("distance_threshold must be positive")
    return [
        nearest_context_distance(context, feature_dim) <= distance_threshold
        for context in contexts
    ]
