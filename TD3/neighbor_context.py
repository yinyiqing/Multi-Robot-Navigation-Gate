import math

import numpy as np


CONTEXT_MODES = ("legacy", "ego_geometry", "ego_motion")


def normalize_context_mode(value):
    mode = (value or "legacy").strip().lower()
    if mode not in CONTEXT_MODES:
        raise ValueError(
            "Unsupported local-Critic context mode: %s. Use %s."
            % (mode, ", ".join(CONTEXT_MODES))
        )
    return mode


def context_feature_dim(mode, legacy_include_actions=False):
    mode = normalize_context_mode(mode)
    if mode == "ego_motion":
        return 7
    if mode == "ego_geometry":
        return 5
    return 7 if legacy_include_actions else 5


def world_to_ego(vector, yaw):
    vector = np.asarray(vector, dtype=np.float32)
    cos_yaw = math.cos(float(yaw))
    sin_yaw = math.sin(float(yaw))
    return np.array(
        [
            cos_yaw * vector[0] + sin_yaw * vector[1],
            -sin_yaw * vector[0] + cos_yaw * vector[1],
        ],
        dtype=np.float32,
    )


def ego_motion_features(offset, ego_yaw, other_yaw, ego_action, other_action):
    ego_offset = world_to_ego(offset, ego_yaw)
    distance = float(np.linalg.norm(offset))
    bearing = math.atan2(float(ego_offset[1]), float(ego_offset[0]))

    ego_linear = float(np.asarray(ego_action, dtype=np.float32)[0])
    other_linear = float(np.asarray(other_action, dtype=np.float32)[0])
    ego_velocity = np.array(
        [ego_linear * math.cos(ego_yaw), ego_linear * math.sin(ego_yaw)],
        dtype=np.float32,
    )
    other_velocity = np.array(
        [other_linear * math.cos(other_yaw), other_linear * math.sin(other_yaw)],
        dtype=np.float32,
    )
    relative_velocity = world_to_ego(other_velocity - ego_velocity, ego_yaw)
    return [
        float(ego_offset[0]),
        float(ego_offset[1]),
        distance,
        float(bearing),
        float(relative_velocity[0]),
        float(relative_velocity[1]),
        1.0,
    ]
