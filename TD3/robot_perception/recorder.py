import os
import re
from pathlib import Path

import numpy as np

from .dataset import build_frame_examples, robot_centers_in_sensor_frame


def _safe_scenario_name(scenario_id):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(scenario_id)).strip("._")
    return value or "unnamed"


class PerceptionShardRecorder:
    """Record one resumable compressed candidate shard per Gazebo scenario."""

    def __init__(
        self,
        output_dir,
        split,
        frame_stride=2,
        max_background_candidates=12,
        actor_state_dim=24,
        oracle_interaction_distance=2.0,
    ):
        if frame_stride < 1:
            raise ValueError("frame_stride must be positive")
        if max_background_candidates < 0:
            raise ValueError("max_background_candidates must be non-negative")
        if actor_state_dim < 1 or oracle_interaction_distance <= 0.0:
            raise ValueError("actor state dim and oracle distance must be positive")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.split = str(split)
        self.frame_stride = int(frame_stride)
        self.max_background_candidates = int(max_background_candidates)
        self.actor_state_dim = int(actor_state_dim)
        self.oracle_interaction_distance = float(oracle_interaction_distance)
        self.current_scenario_id = None
        self.current_path = None
        self.current_pool = "unknown"
        self.current_interaction_band = "unknown"
        self.skip_current = False
        self.frame_index = 0
        self.sampled_frame_count = 0
        self.visible_robot_count = 0
        self.missed_visible_robot_count = 0
        self.records = {}

    def begin_scenario(self, scenario_id, pool="unknown", interaction_band="unknown"):
        if self.current_scenario_id is not None:
            raise RuntimeError("finish the current scenario before starting another")
        self.current_scenario_id = str(scenario_id)
        self.current_pool = str(pool)
        self.current_interaction_band = str(interaction_band)
        self.current_path = self.output_dir / (
            _safe_scenario_name(self.current_scenario_id) + ".npz"
        )
        self.skip_current = self.current_path.exists()
        self.frame_index = 0
        self.sampled_frame_count = 0
        self.visible_robot_count = 0
        self.missed_visible_robot_count = 0
        self.records = {
            "patches": [],
            "labels": [],
            "center_offsets": [],
            "candidate_centers": [],
            "candidate_ranges": [],
            "target_agent_indices": [],
            "ego_poses": [],
            "timestamps": [],
            "ego_indices": [],
            "frame_indices": [],
            "frame_record_indices": [],
            "frame_ego_indices": [],
            "frame_indices_unique": [],
            "frame_ego_poses": [],
            "frame_timestamps": [],
            "frame_actor_states": [],
            "frame_nearest_robot_distances": [],
            "frame_oracle_interaction_labels": [],
            "frame_nearest_front_robot_distances": [],
            "frame_front_interaction_labels": [],
        }
        return not self.skip_current

    def record_frame(
        self,
        raw_lidar_by_agent,
        poses_by_agent,
        active_mask,
        agent_names,
        timestamps_by_agent=None,
        actor_states_by_agent=None,
    ):
        if self.current_scenario_id is None:
            raise RuntimeError("begin_scenario must be called before record_frame")
        current_frame = self.frame_index
        self.frame_index += 1
        if self.skip_current or current_frame % self.frame_stride:
            return
        self.sampled_frame_count += 1
        active_names = [
            name
            for index, name in enumerate(agent_names)
            if index < len(active_mask) and active_mask[index]
        ]
        for ego_index, ego_name in enumerate(agent_names):
            if ego_name not in active_names or ego_name not in poses_by_agent:
                continue
            other_entries = [
                (index, name)
                for index, name in enumerate(agent_names)
                if name in active_names and name != ego_name and name in poses_by_agent
            ]
            other_positions = [
                np.asarray(poses_by_agent[name], dtype=np.float32)[:2]
                for _, name in other_entries
            ]
            ego_pose = np.asarray(poses_by_agent[ego_name], dtype=np.float32)[:3]
            if actor_states_by_agent is None or ego_name not in actor_states_by_agent:
                raise ValueError("actor_states_by_agent must contain every active ego")
            actor_state = np.asarray(
                actor_states_by_agent[ego_name], dtype=np.float32
            ).reshape(-1)
            if actor_state.shape != (self.actor_state_dim,):
                raise ValueError(
                    f"actor state for {ego_name} must have {self.actor_state_dim} values"
                )
            timestamp = (
                float(timestamps_by_agent[ego_name])
                if timestamps_by_agent is not None
                else float(current_frame)
            )
            nearest_robot_distance = (
                min(
                    float(np.linalg.norm(position - ego_pose[:2]))
                    for position in other_positions
                )
                if other_positions
                else np.inf
            )
            sensor_centers = robot_centers_in_sensor_frame(ego_pose, other_positions)
            front_robot_distances = [
                float(np.linalg.norm(position - ego_pose[:2]))
                for position, sensor_center in zip(other_positions, sensor_centers)
                if sensor_center[0] >= 0.0
            ]
            nearest_front_robot_distance = (
                min(front_robot_distances) if front_robot_distances else np.inf
            )
            frame_record_index = len(self.records["frame_ego_indices"])
            self.records["frame_ego_indices"].append(
                np.asarray([ego_index], dtype=np.uint8)
            )
            self.records["frame_indices_unique"].append(
                np.asarray([current_frame], dtype=np.int32)
            )
            self.records["frame_ego_poses"].append(ego_pose[None, :])
            self.records["frame_timestamps"].append(
                np.asarray([timestamp], dtype=np.float64)
            )
            self.records["frame_actor_states"].append(actor_state[None, :])
            self.records["frame_nearest_robot_distances"].append(
                np.asarray([nearest_robot_distance], dtype=np.float32)
            )
            self.records["frame_oracle_interaction_labels"].append(
                np.asarray(
                    [nearest_robot_distance <= self.oracle_interaction_distance],
                    dtype=np.uint8,
                )
            )
            self.records["frame_nearest_front_robot_distances"].append(
                np.asarray([nearest_front_robot_distance], dtype=np.float32)
            )
            self.records["frame_front_interaction_labels"].append(
                np.asarray(
                    [nearest_front_robot_distance <= self.oracle_interaction_distance],
                    dtype=np.uint8,
                )
            )
            examples = build_frame_examples(
                raw_lidar_by_agent.get(ego_name, np.empty((0, 3), dtype=np.float32)),
                ego_pose,
                other_positions,
                other_robot_ids=[index for index, _ in other_entries],
                max_background_candidates=self.max_background_candidates,
            )
            self.visible_robot_count += examples.visible_robot_count
            self.missed_visible_robot_count += examples.missed_visible_robot_count
            count = len(examples.labels)
            if count == 0:
                continue
            self.records["patches"].append(examples.patches.astype(np.float16))
            self.records["labels"].append(examples.labels)
            self.records["center_offsets"].append(
                examples.center_offsets.astype(np.float16)
            )
            self.records["candidate_centers"].append(
                examples.candidate_centers.astype(np.float16)
            )
            self.records["candidate_ranges"].append(
                examples.candidate_ranges.astype(np.float16)
            )
            self.records["target_agent_indices"].append(
                examples.target_agent_indices.astype(np.int16)
            )
            self.records["ego_poses"].append(
                np.repeat(ego_pose[None, :], count, axis=0)
            )
            self.records["timestamps"].append(
                np.full(count, timestamp, dtype=np.float64)
            )
            self.records["ego_indices"].append(
                np.full(count, ego_index, dtype=np.uint8)
            )
            self.records["frame_indices"].append(
                np.full(count, current_frame, dtype=np.int32)
            )
            self.records["frame_record_indices"].append(
                np.full(count, frame_record_index, dtype=np.int32)
            )

    def finish_scenario(self):
        if self.current_scenario_id is None:
            raise RuntimeError("no perception scenario is active")
        path = self.current_path
        if not self.skip_current:
            arrays = {
                "patches": self._concatenate(
                    "patches", (0, 3, 16, 64), np.float16
                ),
                "labels": self._concatenate("labels", (0,), np.uint8),
                "center_offsets": self._concatenate(
                    "center_offsets", (0, 2), np.float16
                ),
                "candidate_centers": self._concatenate(
                    "candidate_centers", (0, 2), np.float16
                ),
                "candidate_ranges": self._concatenate(
                    "candidate_ranges", (0,), np.float16
                ),
                "target_agent_indices": self._concatenate(
                    "target_agent_indices", (0,), np.int16
                ),
                "ego_poses": self._concatenate(
                    "ego_poses", (0, 3), np.float32
                ),
                "timestamps": self._concatenate(
                    "timestamps", (0,), np.float64
                ),
                "ego_indices": self._concatenate("ego_indices", (0,), np.uint8),
                "frame_indices": self._concatenate(
                    "frame_indices", (0,), np.int32
                ),
                "frame_record_indices": self._concatenate(
                    "frame_record_indices", (0,), np.int32
                ),
                "frame_ego_indices": self._concatenate(
                    "frame_ego_indices", (0,), np.uint8
                ),
                "frame_indices_unique": self._concatenate(
                    "frame_indices_unique", (0,), np.int32
                ),
                "frame_ego_poses": self._concatenate(
                    "frame_ego_poses", (0, 3), np.float32
                ),
                "frame_timestamps": self._concatenate(
                    "frame_timestamps", (0,), np.float64
                ),
                "frame_actor_states": self._concatenate(
                    "frame_actor_states", (0, self.actor_state_dim), np.float32
                ),
                "frame_nearest_robot_distances": self._concatenate(
                    "frame_nearest_robot_distances", (0,), np.float32
                ),
                "frame_oracle_interaction_labels": self._concatenate(
                    "frame_oracle_interaction_labels", (0,), np.uint8
                ),
                "frame_nearest_front_robot_distances": self._concatenate(
                    "frame_nearest_front_robot_distances", (0,), np.float32
                ),
                "frame_front_interaction_labels": self._concatenate(
                    "frame_front_interaction_labels", (0,), np.uint8
                ),
                "visible_robot_count": np.asarray(
                    self.visible_robot_count, dtype=np.int64
                ),
                "missed_visible_robot_count": np.asarray(
                    self.missed_visible_robot_count, dtype=np.int64
                ),
                "sampled_frame_count": np.asarray(
                    self.sampled_frame_count, dtype=np.int32
                ),
                "scenario_id": np.asarray(self.current_scenario_id),
                "split": np.asarray(self.split),
                "scenario_pool": np.asarray(self.current_pool),
                "interaction_band": np.asarray(self.current_interaction_band),
                "oracle_interaction_distance": np.asarray(
                    self.oracle_interaction_distance, dtype=np.float32
                ),
                "format_version": np.asarray(3, dtype=np.int32),
            }
            temporary_path = path.with_suffix(path.suffix + ".tmp")
            with temporary_path.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            os.replace(temporary_path, path)

        result = {
            "path": str(path),
            "written": not self.skip_current,
            "candidates": sum(len(item) for item in self.records["labels"]),
            "gate_frames": len(self.records["frame_ego_indices"]),
            "visible_robots": self.visible_robot_count,
            "missed_visible_robots": self.missed_visible_robot_count,
        }
        self.current_scenario_id = None
        self.current_path = None
        self.current_pool = "unknown"
        self.current_interaction_band = "unknown"
        self.records = {}
        return result

    def _concatenate(self, key, empty_shape, dtype):
        values = self.records[key]
        if not values:
            return np.empty(empty_shape, dtype=dtype)
        return np.concatenate(values, axis=0).astype(dtype, copy=False)
