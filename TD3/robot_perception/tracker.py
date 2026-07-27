import math
from dataclasses import dataclass

import numpy as np

from lidar_cluster_tracking import local_to_world


@dataclass(frozen=True)
class TrackedCandidate:
    observation_index: int
    track_id: int
    age: int
    hits: int
    local_center: np.ndarray
    world_center: np.ndarray
    world_velocity: np.ndarray
    relative_velocity: np.ndarray
    shape_probability: float
    smoothed_shape_probability: float
    dynamic_speed: float
    closing_speed: float
    time_to_closest_approach: float
    closest_approach_distance: float
    ttc: float


class RobotCandidateTracker:
    """Associate lidar candidates after compensating for ego motion."""

    def __init__(
        self,
        association_distance=0.65,
        max_missed_frames=2,
        history_size=5,
        probability_alpha=0.35,
        sensor_offset=(0.125, 0.0),
        collision_distance=0.75,
        ttc_horizon=4.0,
    ):
        if association_distance <= 0.0:
            raise ValueError("association_distance must be positive")
        if max_missed_frames < 0 or history_size < 2:
            raise ValueError("tracker lifetime and history are invalid")
        if not 0.0 < probability_alpha <= 1.0:
            raise ValueError("probability_alpha must be within (0, 1]")
        if collision_distance <= 0.0 or ttc_horizon <= 0.0:
            raise ValueError("collision distance and TTC horizon must be positive")
        self.association_distance = float(association_distance)
        self.max_missed_frames = int(max_missed_frames)
        self.history_size = int(history_size)
        self.probability_alpha = float(probability_alpha)
        self.sensor_offset = np.asarray(sensor_offset, dtype=np.float64)
        if self.sensor_offset.shape != (2,):
            raise ValueError("sensor_offset must contain [x, y]")
        self.collision_distance = float(collision_distance)
        self.ttc_horizon = float(ttc_horizon)
        self.reset()

    def reset(self):
        self._tracks = []
        self._next_track_id = 1
        self._previous_timestamp = None
        self._previous_ego_position = None

    @staticmethod
    def _velocity(history):
        if len(history) < 2:
            return np.zeros(2, dtype=np.float64)
        times = np.asarray([item[0] for item in history], dtype=np.float64)
        points = np.asarray([item[1] for item in history], dtype=np.float64)
        centered = times - np.mean(times)
        denominator = float(np.dot(centered, centered))
        if denominator <= 1e-12:
            return np.zeros(2, dtype=np.float64)
        return np.sum(centered[:, None] * points, axis=0) / denominator

    def _risk(self, relative_position, relative_velocity):
        distance = float(np.linalg.norm(relative_position))
        if distance <= 1e-9:
            closing_speed = 0.0
        else:
            closing_speed = max(
                -float(np.dot(relative_position, relative_velocity)) / distance,
                0.0,
            )
        speed_squared = float(np.dot(relative_velocity, relative_velocity))
        if speed_squared <= 1e-12:
            time_to_cpa = self.ttc_horizon
        else:
            time_to_cpa = float(
                np.clip(
                    -np.dot(relative_position, relative_velocity) / speed_squared,
                    0.0,
                    self.ttc_horizon,
                )
            )
        cpa_distance = float(
            np.linalg.norm(relative_position + relative_velocity * time_to_cpa)
        )
        ttc = 0.0 if distance <= self.collision_distance else self.ttc_horizon
        c = distance * distance - self.collision_distance**2
        b = 2.0 * float(np.dot(relative_position, relative_velocity))
        discriminant = b * b - 4.0 * speed_squared * c
        if (
            distance > self.collision_distance
            and speed_squared > 1e-12
            and discriminant >= 0.0
        ):
            roots = [
                (-b - math.sqrt(discriminant)) / (2.0 * speed_squared),
                (-b + math.sqrt(discriminant)) / (2.0 * speed_squared),
            ]
            future = [value for value in roots if value >= 0.0]
            if future:
                ttc = min(min(future), self.ttc_horizon)
        return closing_speed, time_to_cpa, cpa_distance, ttc

    def update(self, candidate_centers, probabilities, ego_pose, timestamp):
        centers = np.asarray(candidate_centers, dtype=np.float64)
        scores = np.asarray(probabilities, dtype=np.float64)
        pose = np.asarray(ego_pose, dtype=np.float64)
        current_time = float(timestamp)
        if centers.ndim != 2 or centers.shape[1:] != (2,):
            raise ValueError("candidate_centers must have shape [N, 2]")
        if scores.shape != (len(centers),) or np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError("probabilities must match candidates and lie within [0, 1]")
        if pose.shape != (3,) or not np.all(np.isfinite(pose)):
            raise ValueError("ego_pose must contain finite [x, y, yaw]")
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(scores)):
            raise ValueError("candidate observations must be finite")
        if self._previous_timestamp is not None and current_time <= self._previous_timestamp:
            raise ValueError("timestamps must increase strictly")

        base_centers = centers + self.sensor_offset[None, :]
        world_centers = np.asarray(
            [local_to_world(center, pose) for center in base_centers],
            dtype=np.float64,
        )
        if self._previous_timestamp is None:
            ego_velocity = np.zeros(2, dtype=np.float64)
        else:
            ego_velocity = (pose[:2] - self._previous_ego_position) / (
                current_time - self._previous_timestamp
            )

        pairs = []
        for observation_index, center in enumerate(world_centers):
            for track_index, track in enumerate(self._tracks):
                distance = float(np.linalg.norm(center - track["world_center"]))
                if distance <= self.association_distance:
                    pairs.append((distance, observation_index, track_index))
        matches = {}
        used_tracks = set()
        for _, observation_index, track_index in sorted(pairs):
            if observation_index in matches or track_index in used_tracks:
                continue
            matches[observation_index] = track_index
            used_tracks.add(track_index)

        next_tracks = []
        outputs = []
        for observation_index, center in enumerate(world_centers):
            track_index = matches.get(observation_index)
            if track_index is None:
                track = {
                    "track_id": self._next_track_id,
                    "age": 1,
                    "hits": 1,
                    "misses": 0,
                    "world_center": center,
                    "history": [(current_time, center)],
                    "shape_probability": float(scores[observation_index]),
                }
                self._next_track_id += 1
            else:
                previous = self._tracks[track_index]
                history = (previous["history"] + [(current_time, center)])[
                    -self.history_size :
                ]
                track = {
                    "track_id": previous["track_id"],
                    "age": previous["age"] + 1,
                    "hits": previous["hits"] + 1,
                    "misses": 0,
                    "world_center": center,
                    "history": history,
                    "shape_probability": (
                        self.probability_alpha * float(scores[observation_index])
                        + (1.0 - self.probability_alpha)
                        * previous["shape_probability"]
                    ),
                }
            next_tracks.append(track)
            world_velocity = self._velocity(track["history"])
            cos_yaw = math.cos(float(pose[2]))
            sin_yaw = math.sin(float(pose[2]))
            relative_world_velocity = world_velocity - ego_velocity
            relative_velocity = np.asarray(
                [
                    cos_yaw * relative_world_velocity[0]
                    + sin_yaw * relative_world_velocity[1],
                    -sin_yaw * relative_world_velocity[0]
                    + cos_yaw * relative_world_velocity[1],
                ]
            )
            closing_speed, time_to_cpa, cpa_distance, ttc = self._risk(
                base_centers[observation_index], relative_velocity
            )
            outputs.append(
                TrackedCandidate(
                    observation_index=observation_index,
                    track_id=track["track_id"],
                    age=track["age"],
                    hits=track["hits"],
                    local_center=centers[observation_index].astype(np.float32),
                    world_center=center.astype(np.float32),
                    world_velocity=world_velocity.astype(np.float32),
                    relative_velocity=relative_velocity.astype(np.float32),
                    shape_probability=float(scores[observation_index]),
                    smoothed_shape_probability=track["shape_probability"],
                    dynamic_speed=float(np.linalg.norm(world_velocity)),
                    closing_speed=closing_speed,
                    time_to_closest_approach=time_to_cpa,
                    closest_approach_distance=cpa_distance,
                    ttc=ttc,
                )
            )

        for track_index, previous in enumerate(self._tracks):
            if track_index in used_tracks:
                continue
            missed = previous["misses"] + 1
            if missed <= self.max_missed_frames:
                next_tracks.append({**previous, "misses": missed})
        self._tracks = next_tracks
        self._previous_timestamp = current_time
        self._previous_ego_position = pose[:2].copy()
        return outputs
