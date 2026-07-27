import os
import re
from pathlib import Path

import numpy as np

from .dataset import build_frame_examples


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
    ):
        if frame_stride < 1:
            raise ValueError("frame_stride must be positive")
        if max_background_candidates < 0:
            raise ValueError("max_background_candidates must be non-negative")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.split = str(split)
        self.frame_stride = int(frame_stride)
        self.max_background_candidates = int(max_background_candidates)
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
            "ego_indices": [],
            "frame_indices": [],
        }
        return not self.skip_current

    def record_frame(self, raw_lidar_by_agent, poses_by_agent, active_mask, agent_names):
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
            other_positions = [
                np.asarray(poses_by_agent[name], dtype=np.float32)[:2]
                for name in active_names
                if name != ego_name and name in poses_by_agent
            ]
            examples = build_frame_examples(
                raw_lidar_by_agent.get(ego_name, np.empty((0, 3), dtype=np.float32)),
                np.asarray(poses_by_agent[ego_name], dtype=np.float32),
                other_positions,
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
            self.records["ego_indices"].append(
                np.full(count, ego_index, dtype=np.uint8)
            )
            self.records["frame_indices"].append(
                np.full(count, current_frame, dtype=np.int32)
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
                "ego_indices": self._concatenate("ego_indices", (0,), np.uint8),
                "frame_indices": self._concatenate(
                    "frame_indices", (0,), np.int32
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
                "format_version": np.asarray(1, dtype=np.int32),
            }
            temporary_path = path.with_suffix(path.suffix + ".tmp")
            with temporary_path.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            os.replace(temporary_path, path)

        result = {
            "path": str(path),
            "written": not self.skip_current,
            "candidates": sum(len(item) for item in self.records["labels"]),
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
