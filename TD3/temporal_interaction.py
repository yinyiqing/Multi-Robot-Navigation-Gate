import math

import numpy as np


def build_front_lidar_gaps(num_sectors):
    if num_sectors < 1:
        raise ValueError("num_sectors must be positive")
    gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / num_sectors]]
    for index in range(num_sectors - 1):
        gaps.append([gaps[index][1], gaps[index][1] + np.pi / num_sectors])
    gaps[-1][-1] += 0.03
    return np.asarray(gaps, dtype=np.float64)


class TemporalInteractionEncoder:
    """Estimate dynamic radial closing after compensating the robot's own motion."""

    def __init__(
        self,
        gaps,
        max_range=10.0,
        collision_distance=0.35,
        max_closing_speed=2.0,
        closing_deadband=0.05,
        ttc_horizon=4.0,
    ):
        self.gaps = np.asarray(gaps, dtype=np.float64)
        if self.gaps.ndim != 2 or self.gaps.shape[1] != 2:
            raise ValueError("gaps must have shape [num_sectors, 2]")
        if np.any(self.gaps[:, 0] >= self.gaps[:, 1]):
            raise ValueError("each lidar gap must satisfy lower < upper")
        if max_range <= 0.0:
            raise ValueError("max_range must be positive")
        if collision_distance < 0.0 or collision_distance >= max_range:
            raise ValueError("collision_distance must be in [0, max_range)")
        if max_closing_speed <= 0.0:
            raise ValueError("max_closing_speed must be positive")
        if closing_deadband < 0.0:
            raise ValueError("closing_deadband must be non-negative")
        if ttc_horizon <= 0.0:
            raise ValueError("ttc_horizon must be positive")

        self.num_sectors = int(self.gaps.shape[0])
        self.sector_angles = np.mean(self.gaps, axis=1)
        self.max_range = float(max_range)
        self.collision_distance = float(collision_distance)
        self.max_closing_speed = float(max_closing_speed)
        self.closing_deadband = float(closing_deadband)
        self.ttc_horizon = float(ttc_horizon)
        self.reset()

    @property
    def feature_dim(self):
        return 2 * self.num_sectors

    def reset(self):
        self.previous_scan = None
        self.previous_pose = None
        self.previous_timestamp = None

    def _validate_scan(self, scan):
        values = np.asarray(scan, dtype=np.float64)
        if values.shape != (self.num_sectors,):
            raise ValueError(
                f"scan must have shape ({self.num_sectors},), got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("scan must contain only finite values")
        return np.clip(values, 0.0, self.max_range)

    @staticmethod
    def _validate_pose(pose):
        values = np.asarray(pose, dtype=np.float64)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError("pose must contain finite [x, y, yaw]")
        return values

    def _sector_index(self, angle):
        matches = np.flatnonzero(
            (self.gaps[:, 0] <= angle) & (angle < self.gaps[:, 1])
        )
        return int(matches[0]) if len(matches) else None

    def _compensate_previous_scan(self, current_pose):
        compensated = np.full(self.num_sectors, np.nan, dtype=np.float64)
        previous_x, previous_y, previous_yaw = self.previous_pose
        current_x, current_y, current_yaw = current_pose
        cos_previous = math.cos(previous_yaw)
        sin_previous = math.sin(previous_yaw)
        cos_current = math.cos(current_yaw)
        sin_current = math.sin(current_yaw)

        for distance, angle in zip(self.previous_scan, self.sector_angles):
            if distance >= self.max_range:
                continue
            local_x = distance * math.cos(angle)
            local_y = distance * math.sin(angle)
            world_x = previous_x + cos_previous * local_x - sin_previous * local_y
            world_y = previous_y + sin_previous * local_x + cos_previous * local_y
            delta_x = world_x - current_x
            delta_y = world_y - current_y
            current_local_x = cos_current * delta_x + sin_current * delta_y
            current_local_y = -sin_current * delta_x + cos_current * delta_y
            projected_angle = math.atan2(current_local_y, current_local_x)
            index = self._sector_index(projected_angle)
            if index is None:
                continue
            projected_distance = math.hypot(current_local_x, current_local_y)
            if np.isnan(compensated[index]) or projected_distance < compensated[index]:
                compensated[index] = projected_distance
        return compensated

    def _empty_result(self):
        zeros = np.zeros(self.num_sectors, dtype=np.float32)
        ttc = np.full(self.num_sectors, self.ttc_horizon, dtype=np.float32)
        return {
            "features": np.concatenate([zeros, zeros]),
            "closing_speed": zeros.copy(),
            "ttc": ttc,
            "urgency": zeros.copy(),
            "valid": np.zeros(self.num_sectors, dtype=bool),
            "summary": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        }

    def update(self, scan, pose, timestamp):
        current_scan = self._validate_scan(scan)
        current_pose = self._validate_pose(pose)
        current_timestamp = float(timestamp)
        if not math.isfinite(current_timestamp):
            raise ValueError("timestamp must be finite")

        if self.previous_scan is None:
            result = self._empty_result()
        else:
            delta_time = current_timestamp - self.previous_timestamp
            if delta_time <= 0.0:
                raise ValueError("timestamps must increase strictly")
            compensated = self._compensate_previous_scan(current_pose)
            valid = np.isfinite(compensated) & (current_scan < self.max_range)
            closing_speed = np.zeros(self.num_sectors, dtype=np.float64)
            closing_speed[valid] = (
                compensated[valid] - current_scan[valid]
            ) / delta_time
            closing_speed = np.clip(
                closing_speed, -self.max_closing_speed, self.max_closing_speed
            )
            approaching = valid & (closing_speed > self.closing_deadband)
            ttc = np.full(self.num_sectors, self.ttc_horizon, dtype=np.float64)
            clearance = np.maximum(current_scan - self.collision_distance, 0.0)
            ttc[approaching] = np.minimum(
                clearance[approaching] / closing_speed[approaching],
                self.ttc_horizon,
            )
            urgency = np.zeros(self.num_sectors, dtype=np.float64)
            urgency[approaching] = 1.0 - ttc[approaching] / self.ttc_horizon
            normalized_closing = np.clip(
                closing_speed / self.max_closing_speed, -1.0, 1.0
            )
            result = {
                "features": np.concatenate([normalized_closing, urgency]).astype(
                    np.float32
                ),
                "closing_speed": closing_speed.astype(np.float32),
                "ttc": ttc.astype(np.float32),
                "urgency": urgency.astype(np.float32),
                "valid": valid,
                "summary": np.array(
                    [
                        float(np.max(urgency)),
                        float(np.min(ttc) / self.ttc_horizon),
                        float(np.mean(urgency)),
                        float(max(np.max(normalized_closing), 0.0)),
                    ],
                    dtype=np.float32,
                ),
            }

        self.previous_scan = current_scan.copy()
        self.previous_pose = current_pose.copy()
        self.previous_timestamp = current_timestamp
        return result
