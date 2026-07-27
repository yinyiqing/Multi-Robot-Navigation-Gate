import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lidar_cluster_tracking import cluster_points, world_to_local

from .range_view import extract_local_patch, project_range_view


@dataclass(frozen=True)
class FrameExamples:
    patches: np.ndarray
    labels: np.ndarray
    center_offsets: np.ndarray
    candidate_centers: np.ndarray
    candidate_ranges: np.ndarray
    target_agent_indices: np.ndarray
    visible_robot_count: int
    missed_visible_robot_count: int


def _empty_examples(vertical_bins=16, patch_width=64, visible_count=0):
    return FrameExamples(
        patches=np.empty((0, 3, vertical_bins, patch_width), dtype=np.float32),
        labels=np.empty((0,), dtype=np.uint8),
        center_offsets=np.empty((0, 2), dtype=np.float32),
        candidate_centers=np.empty((0, 2), dtype=np.float32),
        candidate_ranges=np.empty((0,), dtype=np.float32),
        target_agent_indices=np.empty((0,), dtype=np.int16),
        visible_robot_count=int(visible_count),
        missed_visible_robot_count=int(visible_count),
    )


def robot_centers_in_sensor_frame(
    ego_pose,
    other_world_positions,
    sensor_offset=(0.125, 0.0),
):
    centers = []
    offset = np.asarray(sensor_offset, dtype=np.float32)
    if offset.shape != (2,):
        raise ValueError("sensor_offset must contain [x, y]")
    for position in other_world_positions:
        centers.append(world_to_local(position, ego_pose).astype(np.float32) - offset)
    if not centers:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(centers, dtype=np.float32)


def visible_robot_mask(
    points,
    robot_centers,
    visibility_radius=0.55,
    max_range=6.0,
    horizontal_fov=(-math.pi / 2.0, math.pi / 2.0),
):
    points = np.asarray(points, dtype=np.float32)
    centers = np.asarray(robot_centers, dtype=np.float32)
    if centers.size == 0:
        return np.zeros((0,), dtype=bool)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("points must have shape [N, 2+]")
    result = np.zeros(len(centers), dtype=bool)
    for index, center in enumerate(centers):
        distance = float(np.linalg.norm(center))
        angle = math.atan2(float(center[1]), float(center[0]))
        geometrically_visible = (
            distance <= max_range
            and horizontal_fov[0] <= angle <= horizontal_fov[1]
        )
        if not geometrically_visible or len(points) == 0:
            continue
        nearest_return = float(
            np.min(np.linalg.norm(points[:, :2] - center[None, :], axis=1))
        )
        result[index] = nearest_return <= visibility_radius
    return result


def match_candidates(candidate_centers, robot_centers, max_distance=0.6):
    candidates = np.asarray(candidate_centers, dtype=np.float32)
    robots = np.asarray(robot_centers, dtype=np.float32)
    matches = {}
    pairs = []
    for candidate_index, candidate in enumerate(candidates):
        for robot_index, robot in enumerate(robots):
            distance = float(np.linalg.norm(candidate - robot))
            if distance <= max_distance:
                pairs.append((distance, candidate_index, robot_index))
    used_robots = set()
    for _, candidate_index, robot_index in sorted(pairs):
        if candidate_index in matches or robot_index in used_robots:
            continue
        matches[candidate_index] = robot_index
        used_robots.add(robot_index)
    return matches


def build_frame_examples(
    points,
    ego_pose,
    other_world_positions,
    other_robot_ids=None,
    max_candidate_range=4.0,
    max_sensor_range=6.0,
    physical_width=1.2,
    patch_width=64,
    max_background_candidates=12,
    visibility_radius=0.55,
    match_distance=0.6,
    cluster_kwargs=None,
):
    """Create deployable candidate patches and simulator-only identity labels."""
    values = np.asarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError("points must have shape [N, 3+]")
    if max_background_candidates < 0:
        raise ValueError("max_background_candidates must be non-negative")
    robot_centers = robot_centers_in_sensor_frame(ego_pose, other_world_positions)
    if other_robot_ids is None:
        robot_ids = np.arange(len(robot_centers), dtype=np.int16)
    else:
        robot_ids = np.asarray(other_robot_ids, dtype=np.int16)
        if robot_ids.shape != (len(robot_centers),):
            raise ValueError("other_robot_ids must match other_world_positions")
    visible_mask = visible_robot_mask(
        values,
        robot_centers,
        visibility_radius=visibility_radius,
        max_range=max_candidate_range,
    )
    visible_centers = robot_centers[visible_mask]
    visible_robot_ids = robot_ids[visible_mask]
    visible_count = len(visible_centers)

    kwargs = {
        "connection_distance": 0.14,
        "min_points": 3,
        "min_diameter": 0.05,
        "max_diameter": 0.9,
    }
    if cluster_kwargs:
        kwargs.update(cluster_kwargs)
    clusters = [
        cluster
        for cluster in cluster_points(values, **kwargs)
        if 0.2 <= float(np.linalg.norm(cluster["centroid"])) <= max_candidate_range
    ]
    if not clusters:
        return _empty_examples(16, patch_width, visible_count)

    centers = np.asarray([cluster["centroid"] for cluster in clusters], dtype=np.float32)
    matches = match_candidates(centers, visible_centers, max_distance=match_distance)
    positive_indices = sorted(matches)
    background_indices = sorted(
        (index for index in range(len(centers)) if index not in matches),
        key=lambda index: float(np.linalg.norm(centers[index])),
    )[:max_background_candidates]
    selected_indices = positive_indices + background_indices
    if not selected_indices:
        return _empty_examples(16, patch_width, visible_count)

    view = project_range_view(values, max_range=max_sensor_range)
    patches = []
    labels = []
    offsets = []
    selected_centers = []
    target_agent_indices = []
    for candidate_index in selected_indices:
        center = centers[candidate_index]
        patches.append(
            extract_local_patch(
                view,
                center,
                physical_width=physical_width,
                output_width=patch_width,
            )
        )
        robot_index = matches.get(candidate_index)
        is_robot = robot_index is not None
        labels.append(int(is_robot))
        offsets.append(
            visible_centers[robot_index] - center
            if is_robot
            else np.zeros(2, dtype=np.float32)
        )
        target_agent_indices.append(
            int(visible_robot_ids[robot_index]) if is_robot else -1
        )
        selected_centers.append(center)

    labels_array = np.asarray(labels, dtype=np.uint8)
    return FrameExamples(
        patches=np.asarray(patches, dtype=np.float32),
        labels=labels_array,
        center_offsets=np.asarray(offsets, dtype=np.float32),
        candidate_centers=np.asarray(selected_centers, dtype=np.float32),
        candidate_ranges=np.linalg.norm(selected_centers, axis=1).astype(np.float32),
        target_agent_indices=np.asarray(target_agent_indices, dtype=np.int16),
        visible_robot_count=int(visible_count),
        missed_visible_robot_count=max(int(visible_count - labels_array.sum()), 0),
    )


REQUIRED_SHARD_KEYS = {
    "patches",
    "labels",
    "center_offsets",
    "candidate_centers",
    "candidate_ranges",
    "visible_robot_count",
    "missed_visible_robot_count",
}


def list_shards(directory):
    paths = sorted(Path(directory).glob("*.npz"))
    if not paths:
        raise ValueError("no perception shards found in %s" % directory)
    return paths


def load_shard(path):
    with np.load(path, allow_pickle=False) as payload:
        missing = REQUIRED_SHARD_KEYS - set(payload.files)
        if missing:
            raise ValueError("shard %s is missing keys: %s" % (path, sorted(missing)))
        return {key: payload[key] for key in payload.files}
