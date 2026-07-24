import math

import numpy as np


def _validate_points(points):
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("points must have shape [N, 2+] ")
    if not np.all(np.isfinite(values)):
        raise ValueError("points must be finite")
    return values[:, :2]


def cluster_points(
    points,
    connection_distance=0.14,
    min_points=3,
    min_diameter=0.05,
    max_diameter=0.9,
):
    """Group voxelized 2D lidar points without requiring a DBSCAN dependency."""
    values = _validate_points(points)
    if connection_distance <= 0.0:
        raise ValueError("connection_distance must be positive")
    if min_points < 1:
        raise ValueError("min_points must be positive")
    if min_diameter < 0.0 or max_diameter <= min_diameter:
        raise ValueError("cluster diameter bounds are invalid")
    if len(values) == 0:
        return []

    cell_size = float(connection_distance)
    cells = np.floor(values / cell_size).astype(np.int64)
    cell_members = {}
    for index, cell in enumerate(cells):
        cell_members.setdefault((int(cell[0]), int(cell[1])), []).append(index)

    visited = np.zeros(len(values), dtype=bool)
    clusters = []
    distance_squared = connection_distance ** 2
    for seed in range(len(values)):
        if visited[seed]:
            continue
        visited[seed] = True
        component = []
        queue = [seed]
        while queue:
            current = queue.pop()
            component.append(current)
            cell_x, cell_y = cells[current]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for candidate in cell_members.get(
                        (int(cell_x + dx), int(cell_y + dy)), ()
                    ):
                        if visited[candidate]:
                            continue
                        difference = values[candidate] - values[current]
                        if float(np.dot(difference, difference)) <= distance_squared:
                            visited[candidate] = True
                            queue.append(candidate)

        if len(component) < min_points:
            continue
        component_points = values[component]
        extent = np.ptp(component_points, axis=0)
        diameter = float(np.linalg.norm(extent))
        if diameter < min_diameter or diameter > max_diameter:
            continue
        centroid = np.mean(component_points, axis=0)
        clusters.append(
            {
                "centroid": centroid,
                "diameter": diameter,
                "point_count": len(component),
                "indices": np.asarray(component, dtype=np.int64),
            }
        )
    return clusters


def local_to_world(point, pose):
    x, y, yaw = np.asarray(pose, dtype=np.float64)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    local_x, local_y = np.asarray(point, dtype=np.float64)
    return np.array(
        [x + cos_yaw * local_x - sin_yaw * local_y,
         y + sin_yaw * local_x + cos_yaw * local_y],
        dtype=np.float64,
    )


def world_to_local(point, pose):
    x, y, yaw = np.asarray(pose, dtype=np.float64)
    delta_x, delta_y = np.asarray(point, dtype=np.float64) - np.array([x, y])
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return np.array(
        [cos_yaw * delta_x + sin_yaw * delta_y,
         -sin_yaw * delta_x + cos_yaw * delta_y],
        dtype=np.float64,
    )


class LidarClusterTracker:
    """Track compact lidar clusters and estimate deployable CPA/TTC features."""

    def __init__(
        self,
        connection_distance=0.14,
        min_points=3,
        min_diameter=0.05,
        max_diameter=0.9,
        association_distance=0.65,
        max_delta_time=0.5,
        collision_distance=0.75,
        ttc_horizon=4.0,
        closing_deadband=0.05,
        dynamic_speed_deadband=0.1,
    ):
        if association_distance <= 0.0:
            raise ValueError("association_distance must be positive")
        if max_delta_time <= 0.0:
            raise ValueError("max_delta_time must be positive")
        if collision_distance <= 0.0:
            raise ValueError("collision_distance must be positive")
        if ttc_horizon <= 0.0:
            raise ValueError("ttc_horizon must be positive")
        self.cluster_kwargs = {
            "connection_distance": connection_distance,
            "min_points": min_points,
            "min_diameter": min_diameter,
            "max_diameter": max_diameter,
        }
        self.association_distance = float(association_distance)
        self.max_delta_time = float(max_delta_time)
        self.collision_distance = float(collision_distance)
        self.ttc_horizon = float(ttc_horizon)
        self.closing_deadband = float(closing_deadband)
        self.dynamic_speed_deadband = float(dynamic_speed_deadband)
        self.reset()

    def reset(self):
        self.previous_clusters = []
        self.previous_pose = None
        self.previous_timestamp = None
        self.next_track_id = 1

    @staticmethod
    def _validate_pose(pose):
        values = np.asarray(pose, dtype=np.float64)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError("pose must contain finite [x, y, yaw]")
        return values

    def _new_tracks(self, clusters, current_pose, current_timestamp):
        tracks = []
        for cluster in clusters:
            tracks.append(
                {
                    **cluster,
                    "track_id": self.next_track_id,
                    "age": 1,
                    "velocity": np.zeros(2, dtype=np.float64),
                    "world_centroid": local_to_world(
                        cluster["centroid"], current_pose
                    ),
                    "world_velocity": np.zeros(2, dtype=np.float64),
                    "world_history": [
                        (
                            current_timestamp,
                            local_to_world(cluster["centroid"], current_pose),
                        )
                    ],
                    "closing_speed": 0.0,
                    "time_to_closest_approach": self.ttc_horizon,
                    "closest_approach_distance": float(
                        np.linalg.norm(cluster["centroid"])
                    ),
                    "ttc": self.ttc_horizon,
                    "urgent": False,
                }
            )
            self.next_track_id += 1
        return tracks

    def update(self, points, pose, timestamp):
        current_pose = self._validate_pose(pose)
        current_timestamp = float(timestamp)
        if not math.isfinite(current_timestamp):
            raise ValueError("timestamp must be finite")
        clusters = cluster_points(points, **self.cluster_kwargs)

        if self.previous_timestamp is None:
            tracks = self._new_tracks(clusters, current_pose, current_timestamp)
        else:
            delta_time = current_timestamp - self.previous_timestamp
            if delta_time <= 0.0:
                raise ValueError("timestamps must increase strictly")
            if delta_time > self.max_delta_time:
                tracks = self._new_tracks(clusters, current_pose, current_timestamp)
            else:
                projected_previous = [
                    world_to_local(
                        local_to_world(item["centroid"], self.previous_pose),
                        current_pose,
                    )
                    for item in self.previous_clusters
                ]
                pairs = []
                for current_index, current in enumerate(clusters):
                    for previous_index, previous in enumerate(self.previous_clusters):
                        if abs(current["diameter"] - previous["diameter"]) > 0.45:
                            continue
                        distance = float(
                            np.linalg.norm(
                                current["centroid"]
                                - projected_previous[previous_index]
                            )
                        )
                        if distance <= self.association_distance:
                            pairs.append((distance, current_index, previous_index))
                pairs.sort()
                matches = {}
                used_previous = set()
                for _, current_index, previous_index in pairs:
                    if current_index in matches or previous_index in used_previous:
                        continue
                    matches[current_index] = previous_index
                    used_previous.add(previous_index)

                tracks = []
                for current_index, cluster in enumerate(clusters):
                    previous_index = matches.get(current_index)
                    if previous_index is None:
                        tracks.extend(
                            self._new_tracks(
                                [cluster], current_pose, current_timestamp
                            )
                        )
                        continue
                    previous = self.previous_clusters[previous_index]
                    current_world_centroid = local_to_world(
                        cluster["centroid"], current_pose
                    )
                    world_history = previous["world_history"] + [
                        (current_timestamp, current_world_centroid)
                    ]
                    world_history = world_history[-5:]
                    history_times = np.asarray(
                        [item[0] for item in world_history], dtype=np.float64
                    )
                    history_points = np.asarray(
                        [item[1] for item in world_history], dtype=np.float64
                    )
                    centered_times = history_times - np.mean(history_times)
                    denominator = float(np.dot(centered_times, centered_times))
                    world_velocity = (
                        np.sum(
                            centered_times[:, None]
                            * (history_points - np.mean(history_points, axis=0)),
                            axis=0,
                        )
                        / denominator
                        if denominator > 1e-8
                        else np.zeros(2, dtype=np.float64)
                    )
                    ego_world_velocity = (
                        current_pose[:2] - self.previous_pose[:2]
                    ) / delta_time
                    relative_world_velocity = world_velocity - ego_world_velocity
                    cos_yaw = math.cos(current_pose[2])
                    sin_yaw = math.sin(current_pose[2])
                    velocity = np.array(
                        [
                            cos_yaw * relative_world_velocity[0]
                            + sin_yaw * relative_world_velocity[1],
                            -sin_yaw * relative_world_velocity[0]
                            + cos_yaw * relative_world_velocity[1],
                        ],
                        dtype=np.float64,
                    )
                    position = cluster["centroid"]
                    distance = float(np.linalg.norm(position))
                    closing_speed = (
                        -float(np.dot(position, velocity)) / distance
                        if distance > 1e-6
                        else 0.0
                    )
                    speed_squared = float(np.dot(velocity, velocity))
                    time_to_closest = (
                        max(-float(np.dot(position, velocity)) / speed_squared, 0.0)
                        if speed_squared > 1e-8
                        else self.ttc_horizon
                    )
                    closest_position = position + velocity * time_to_closest
                    closest_distance = float(np.linalg.norm(closest_position))
                    urgent = bool(
                        np.linalg.norm(world_velocity) > self.dynamic_speed_deadband
                        and
                        closing_speed > self.closing_deadband
                        and time_to_closest <= self.ttc_horizon
                        and closest_distance <= self.collision_distance
                    )
                    tracks.append(
                        {
                            **cluster,
                            "track_id": previous["track_id"],
                            "age": previous["age"] + 1,
                            "velocity": velocity,
                            "world_centroid": current_world_centroid,
                            "world_velocity": world_velocity,
                            "world_history": world_history,
                            "closing_speed": closing_speed,
                            "time_to_closest_approach": min(
                                time_to_closest, self.ttc_horizon
                            ),
                            "closest_approach_distance": closest_distance,
                            "ttc": min(time_to_closest, self.ttc_horizon)
                            if urgent
                            else self.ttc_horizon,
                            "urgent": urgent,
                        }
                    )

        self.previous_clusters = [
            {
                "centroid": item["centroid"].copy(),
                "diameter": item["diameter"],
                "point_count": item["point_count"],
                "track_id": item["track_id"],
                "age": item["age"],
                "world_centroid": item["world_centroid"].copy(),
                "world_velocity": item["world_velocity"].copy(),
                "world_history": [
                    (timestamp, point.copy())
                    for timestamp, point in item["world_history"]
                ],
            }
            for item in tracks
        ]
        self.previous_pose = current_pose.copy()
        self.previous_timestamp = current_timestamp
        return tracks
