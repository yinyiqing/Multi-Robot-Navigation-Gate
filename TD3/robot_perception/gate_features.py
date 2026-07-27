import math

import numpy as np


TRACK_FEATURE_DIM = 11
GLOBAL_FEATURE_DIM = 8


def _track_priority(item):
    range_m = float(np.linalg.norm(item.local_center))
    return (
        max(item.shape_probability, item.smoothed_shape_probability)
        + 0.25 * min(item.dynamic_speed, 1.0)
        + 0.25 * min(item.closing_speed, 1.0)
        - 0.05 * min(range_m, 4.0)
    )


def summarize_tracked_candidates(tracked_candidates, max_tracks=4):
    if max_tracks < 1:
        raise ValueError("max_tracks must be positive")
    candidates = sorted(tracked_candidates, key=_track_priority, reverse=True)
    result = np.zeros(
        max_tracks * TRACK_FEATURE_DIM + GLOBAL_FEATURE_DIM, dtype=np.float32
    )
    ranges = []
    raw_scores = []
    smoothed_scores = []
    dynamic_speeds = []
    closing_speeds = []
    ttcs = []
    for index, item in enumerate(candidates[:max_tracks]):
        center = np.asarray(item.local_center, dtype=np.float32)
        range_m = float(np.linalg.norm(center))
        bearing = math.atan2(float(center[1]), float(center[0]))
        start = index * TRACK_FEATURE_DIM
        result[start : start + TRACK_FEATURE_DIM] = np.asarray(
            [
                1.0,
                item.shape_probability,
                item.smoothed_shape_probability,
                np.clip(range_m / 4.0, 0.0, 1.5),
                math.sin(bearing),
                math.cos(bearing),
                np.clip(item.dynamic_speed / 1.5, 0.0, 2.0),
                np.clip(item.closing_speed / 1.5, 0.0, 2.0),
                np.clip(item.ttc / 4.0, 0.0, 1.0),
                np.clip(item.closest_approach_distance / 4.0, 0.0, 1.5),
                np.clip(item.age / 5.0, 0.0, 2.0),
            ],
            dtype=np.float32,
        )
        ranges.append(range_m)
        raw_scores.append(float(item.shape_probability))
        smoothed_scores.append(float(item.smoothed_shape_probability))
        dynamic_speeds.append(float(item.dynamic_speed))
        closing_speeds.append(float(item.closing_speed))
        ttcs.append(float(item.ttc))
    if candidates:
        global_start = max_tracks * TRACK_FEATURE_DIM
        result[global_start:] = np.asarray(
            [
                min(len(candidates) / 16.0, 1.5),
                1.0,
                max(raw_scores),
                max(smoothed_scores),
                min(min(ranges) / 4.0, 1.5),
                min(max(dynamic_speeds) / 1.5, 2.0),
                min(max(closing_speeds) / 1.5, 2.0),
                min(min(ttcs) / 4.0, 1.0),
            ],
            dtype=np.float32,
        )
    return result


def build_gate_feature(actor_state, tracked_candidates, max_tracks=4):
    state = np.asarray(actor_state, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(state)):
        raise ValueError("actor_state must be finite")
    return np.concatenate(
        (state, summarize_tracked_candidates(tracked_candidates, max_tracks))
    ).astype(np.float32)


def gate_feature_dim(actor_state_dim=24, max_tracks=4):
    if actor_state_dim < 1 or max_tracks < 1:
        raise ValueError("feature dimensions must be positive")
    return actor_state_dim + max_tracks * TRACK_FEATURE_DIM + GLOBAL_FEATURE_DIM
