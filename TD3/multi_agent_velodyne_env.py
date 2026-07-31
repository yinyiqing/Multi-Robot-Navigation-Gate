import math
import json
import os
import random
import socket
import subprocess
import time
from os import path

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetPhysicsProperties, GetWorldProperties, SetModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from scenario_manifests import load_manifest_dataset, validate_manifest_scenarios
from temporal_interaction import build_front_lidar_gaps
from velodyne_env import COLLISION_DIST, GOAL_REACHED_DIST, TIME_DELTA, check_pos
from neighbor_context import (
    context_feature_dim,
    ego_motion_features,
    normalize_context_mode,
    world_to_ego,
)


AGENT_COLORS = [
    (1.0, 0.20, 0.12),
    (0.15, 0.55, 1.0),
    (0.10, 0.95, 0.35),
    (1.0, 0.75, 0.10),
    (0.85, 0.35, 1.0),
    (0.10, 0.95, 0.95),
    (1.0, 0.45, 0.55),
    (0.55, 0.85, 0.10),
    (0.70, 0.70, 1.0),
    (1.0, 0.55, 0.10),
]


def _env_float(name, default):
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return bool(default)
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean value")


def _env_range(name, default):
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"{name} must be formatted as min,max")
    low, high = float(parts[0]), float(parts[1])
    if low >= high:
        raise ValueError(f"{name} must satisfy min < max")
    return (low, high)


class MultiAgentGazeboEnv:
    """Gazebo environment with multiple robots sharing the same policy."""

    def __init__(
        self,
        launchfile,
        environment_dim,
        agent_names=None,
        cooperative_reward=False,
        cooperative_reward_self_weight=None,
        cooperative_reward_distance_weighted=False,
        cooperative_reward_sigma=2.0,
        cooperative_reward_mode="average",
        interaction_safe_distance=1.2,
        interaction_close_penalty=0.5,
        interaction_stagnation_penalty=0.05,
        anti_stagnation_reward=False,
        anti_stagnation_penalty=0.2,
        anti_stagnation_linear_threshold=0.05,
        anti_stagnation_progress_threshold=0.005,
        anti_stagnation_min_laser=0.35,
        safe_recovery_reward=False,
        safe_recovery_penalty=0.2,
        safe_recovery_linear_threshold=0.25,
        safe_recovery_progress_threshold=0.003,
        safe_recovery_min_laser=0.6,
        safe_recovery_robot_distance=1.2,
        safe_recovery_progress_bonus_weight=0.0,
        safe_recovery_idle_penalty_weight=0.0,
        wall_clearance_reward=False,
        wall_clearance_safe_distance=0.75,
        wall_clearance_penalty=1.5,
        wall_clearance_speed_weight=0.8,
        wall_clearance_turn_weight=0.4,
        local_navigation_reward=False,
        local_navigation_heading_weight=0.4,
        local_navigation_wrong_way_penalty=0.25,
        local_navigation_turn_weight=0.25,
        local_navigation_near_goal_distance=0.9,
        local_navigation_heading_error=0.5,
        reward_neighbor_radius=10.0,
        reward_neighbor_fov=np.pi / 2 + 0.03,
        robot_safe_distance=1.0,
        robot_proximity_penalty_weight=5.0,
        robot_proximity_speed_penalty_weight=0.0,
        robot_clearance_reward_weight=0.0,
        robot_clearance_reward_max_gain=0.1,
        yield_priority_reward=False,
        yield_priority_distance=1.0,
        yield_priority_goal_margin=0.25,
        yield_priority_stop_linear=0.25,
        yield_priority_penalty_weight=4.0,
        yield_priority_bonus_weight=1.0,
        yield_priority_clearance_bonus_weight=2.0,
        yield_priority_restart_bonus_weight=1.5,
        yield_priority_stale_wait_penalty_weight=0.5,
        yield_priority_max_wait_steps=8,
        emergency_stop_distance=0.6,
        emergency_stop_penalty_weight=8.0,
        emergency_stop_bonus_weight=1.0,
        forward_reward_weight=0.5,
        stagnation_penalty_weight=0.03,
        weak_coupling_layout=False,
        scenario_mode="standard",
        active_neighbors_only=False,
        neighbor_context_mode="legacy",
        fixed_physics_step_size=None,
    ):
        self.environment_dim = environment_dim
        self.agent_names = agent_names or ["r1", "r2", "r3"]
        self.num_agents = len(self.agent_names)
        self.cooperative_reward = cooperative_reward
        self.cooperative_reward_self_weight = cooperative_reward_self_weight
        self.cooperative_reward_distance_weighted = cooperative_reward_distance_weighted
        self.cooperative_reward_sigma = cooperative_reward_sigma
        self.cooperative_reward_mode = cooperative_reward_mode.strip().lower()
        if self.cooperative_reward_mode not in (
            "average",
            "average_plus_interaction",
            "interaction_only",
        ):
            raise ValueError(
                "cooperative_reward_mode must be 'average', "
                "'average_plus_interaction', or 'interaction_only'"
            )
        self.interaction_safe_distance = interaction_safe_distance
        self.interaction_close_penalty = interaction_close_penalty
        self.interaction_stagnation_penalty = interaction_stagnation_penalty
        self.anti_stagnation_reward = anti_stagnation_reward
        self.anti_stagnation_penalty = anti_stagnation_penalty
        self.anti_stagnation_linear_threshold = anti_stagnation_linear_threshold
        self.anti_stagnation_progress_threshold = anti_stagnation_progress_threshold
        self.anti_stagnation_min_laser = anti_stagnation_min_laser
        self.safe_recovery_reward = bool(safe_recovery_reward)
        self.safe_recovery_penalty = float(safe_recovery_penalty)
        self.safe_recovery_linear_threshold = float(safe_recovery_linear_threshold)
        self.safe_recovery_progress_threshold = float(safe_recovery_progress_threshold)
        self.safe_recovery_min_laser = float(safe_recovery_min_laser)
        self.safe_recovery_robot_distance = float(safe_recovery_robot_distance)
        self.safe_recovery_progress_bonus_weight = float(
            safe_recovery_progress_bonus_weight
        )
        self.safe_recovery_idle_penalty_weight = float(
            safe_recovery_idle_penalty_weight
        )
        self.wall_clearance_reward = wall_clearance_reward
        self.wall_clearance_safe_distance = wall_clearance_safe_distance
        self.wall_clearance_penalty = wall_clearance_penalty
        self.wall_clearance_speed_weight = wall_clearance_speed_weight
        self.wall_clearance_turn_weight = wall_clearance_turn_weight
        self.local_navigation_reward = local_navigation_reward
        self.local_navigation_heading_weight = local_navigation_heading_weight
        self.local_navigation_wrong_way_penalty = local_navigation_wrong_way_penalty
        self.local_navigation_turn_weight = local_navigation_turn_weight
        self.local_navigation_near_goal_distance = local_navigation_near_goal_distance
        self.local_navigation_heading_error = local_navigation_heading_error
        self.reward_neighbor_radius = reward_neighbor_radius
        self.reward_neighbor_fov = reward_neighbor_fov
        self.robot_safe_distance = float(robot_safe_distance)
        self.robot_proximity_penalty_weight = float(
            robot_proximity_penalty_weight
        )
        self.robot_proximity_speed_penalty_weight = float(
            robot_proximity_speed_penalty_weight
        )
        self.robot_clearance_reward_weight = float(robot_clearance_reward_weight)
        self.robot_clearance_reward_max_gain = float(
            robot_clearance_reward_max_gain
        )
        self.yield_priority_reward = bool(yield_priority_reward)
        self.yield_priority_distance = float(yield_priority_distance)
        self.yield_priority_goal_margin = float(yield_priority_goal_margin)
        self.yield_priority_stop_linear = float(yield_priority_stop_linear)
        self.yield_priority_penalty_weight = float(yield_priority_penalty_weight)
        self.yield_priority_bonus_weight = float(yield_priority_bonus_weight)
        self.yield_priority_clearance_bonus_weight = float(
            yield_priority_clearance_bonus_weight
        )
        self.yield_priority_restart_bonus_weight = float(
            yield_priority_restart_bonus_weight
        )
        self.yield_priority_stale_wait_penalty_weight = float(
            yield_priority_stale_wait_penalty_weight
        )
        self.yield_priority_max_wait_steps = int(yield_priority_max_wait_steps)
        self.emergency_stop_distance = float(emergency_stop_distance)
        self.emergency_stop_penalty_weight = float(emergency_stop_penalty_weight)
        self.emergency_stop_bonus_weight = float(emergency_stop_bonus_weight)
        self.forward_reward_weight = float(forward_reward_weight)
        self.stagnation_penalty_weight = float(stagnation_penalty_weight)
        if self.forward_reward_weight < 0.0:
            raise ValueError("forward_reward_weight must be non-negative")
        if self.stagnation_penalty_weight < 0.0:
            raise ValueError("stagnation_penalty_weight must be non-negative")
        for name, value in (
            ("safe_recovery_penalty", self.safe_recovery_penalty),
            ("safe_recovery_linear_threshold", self.safe_recovery_linear_threshold),
            (
                "safe_recovery_progress_threshold",
                self.safe_recovery_progress_threshold,
            ),
            ("safe_recovery_min_laser", self.safe_recovery_min_laser),
            ("safe_recovery_robot_distance", self.safe_recovery_robot_distance),
            (
                "safe_recovery_progress_bonus_weight",
                self.safe_recovery_progress_bonus_weight,
            ),
            ("safe_recovery_idle_penalty_weight", self.safe_recovery_idle_penalty_weight),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.robot_safe_distance < 0.0:
            raise ValueError("robot_safe_distance must be non-negative")
        if self.robot_proximity_penalty_weight < 0.0:
            raise ValueError("robot_proximity_penalty_weight must be non-negative")
        if self.robot_proximity_speed_penalty_weight < 0.0:
            raise ValueError(
                "robot_proximity_speed_penalty_weight must be non-negative"
            )
        if self.robot_clearance_reward_weight < 0.0:
            raise ValueError("robot_clearance_reward_weight must be non-negative")
        if self.robot_clearance_reward_max_gain <= 0.0:
            raise ValueError("robot_clearance_reward_max_gain must be positive")
        if self.yield_priority_distance <= 0.0:
            raise ValueError("yield_priority_distance must be positive")
        if self.yield_priority_goal_margin < 0.0:
            raise ValueError("yield_priority_goal_margin must be non-negative")
        if not 0.0 <= self.yield_priority_stop_linear <= 1.0:
            raise ValueError("yield_priority_stop_linear must be in [0, 1]")
        for name, value in (
            ("yield_priority_penalty_weight", self.yield_priority_penalty_weight),
            ("yield_priority_bonus_weight", self.yield_priority_bonus_weight),
            (
                "yield_priority_clearance_bonus_weight",
                self.yield_priority_clearance_bonus_weight,
            ),
            (
                "yield_priority_restart_bonus_weight",
                self.yield_priority_restart_bonus_weight,
            ),
            (
                "yield_priority_stale_wait_penalty_weight",
                self.yield_priority_stale_wait_penalty_weight,
            ),
            ("emergency_stop_penalty_weight", self.emergency_stop_penalty_weight),
            ("emergency_stop_bonus_weight", self.emergency_stop_bonus_weight),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.yield_priority_max_wait_steps < 0:
            raise ValueError("yield_priority_max_wait_steps must be non-negative")
        if self.emergency_stop_distance < 0.0:
            raise ValueError("emergency_stop_distance must be non-negative")
        self.weak_coupling_layout = weak_coupling_layout
        self.active_neighbors_only = active_neighbors_only
        self.neighbor_context_mode = normalize_context_mode(neighbor_context_mode)
        self.fixed_physics_step_size = (
            None
            if fixed_physics_step_size is None
            else float(fixed_physics_step_size)
        )
        if (
            self.fixed_physics_step_size is not None
            and self.fixed_physics_step_size <= 0.0
        ):
            raise ValueError("fixed_physics_step_size must be positive")
        self.scenario_mode = scenario_mode.strip().lower()
        if self.scenario_mode not in ("standard", "dense", "curriculum", "manifest"):
            raise ValueError(
                "scenario_mode must be one of: standard, dense, curriculum, manifest"
            )
        self.curriculum_cases = []
        self.curriculum_case_index = 0
        self.manifest_band_cases = {}
        self.manifest_band_indices = {}
        self.manifest_band_schedule_index = 0
        self.current_curriculum_case = None
        self.roscore_process = None
        self.roslaunch_process = None

        self.upper = 5.0
        self.lower = -5.0
        self.goal_min_distance = 0.8
        self.goal_clearance = 1.2
        self.goal_max_distance = 5.0
        self.capacity_robot_clearance = _env_float(
            "DRL_MULTI_CAPACITY_ROBOT_CLEARANCE", 0.95
        )
        self.capacity_goal_clearance = _env_float(
            "DRL_MULTI_CAPACITY_GOAL_CLEARANCE", 0.85
        )
        self.capacity_robot_goal_clearance = _env_float(
            "DRL_MULTI_CAPACITY_ROBOT_GOAL_CLEARANCE", 0.75
        )
        self.capacity_goal_max_distance = _env_float(
            "DRL_MULTI_CAPACITY_GOAL_MAX_DISTANCE", 3.5
        )
        self.capacity_goal_x_offset = _env_range(
            "DRL_MULTI_CAPACITY_GOAL_X_OFFSET", (-2.2, 2.2)
        )
        self.capacity_goal_y_offset = _env_range(
            "DRL_MULTI_CAPACITY_GOAL_Y_OFFSET", (-2.4, 2.4)
        )
        self.dense_start_x_range = _env_range(
            "DRL_MULTI_DENSE_START_X_RANGE", (-2.0, 2.0)
        )
        self.dense_start_y_range = _env_range(
            "DRL_MULTI_DENSE_START_Y_RANGE", (-2.0, 2.0)
        )
        self.dense_goal_x_offset = _env_range(
            "DRL_MULTI_DENSE_GOAL_X_OFFSET", (-1.2, 1.2)
        )
        self.dense_goal_y_offset = _env_range(
            "DRL_MULTI_DENSE_GOAL_Y_OFFSET", (-1.2, 1.2)
        )
        self.dense_robot_clearance = _env_float(
            "DRL_MULTI_DENSE_ROBOT_CLEARANCE", 0.9
        )
        self.dense_goal_clearance = _env_float("DRL_MULTI_DENSE_GOAL_CLEARANCE", 0.8)
        self.dense_goal_min_distance = _env_float(
            "DRL_MULTI_DENSE_GOAL_MIN_DISTANCE", 0.6
        )
        self.dense_goal_max_distance = _env_float(
            "DRL_MULTI_DENSE_GOAL_MAX_DISTANCE", 2.5
        )
        if self.scenario_mode == "curriculum":
            self.curriculum_cases = self._load_curriculum_cases()
            print(
                "Curriculum scenario enabled: %i cases loaded"
                % len(self.curriculum_cases)
            )
        elif self.scenario_mode == "manifest":
            self.curriculum_cases = self._load_manifest_cases()
            self._reset_manifest_band_sampling()
            print("Fixed manifest enabled: %i scenarios loaded" % len(self.curriculum_cases))

        self.velodyne_data = {
            name: np.ones(self.environment_dim) * 10 for name in self.agent_names
        }
        self.temporal_lidar_dim = _env_int("DRL_MULTI_TEMPORAL_LIDAR_DIM", 0)
        if self.temporal_lidar_dim < 0:
            raise ValueError("DRL_MULTI_TEMPORAL_LIDAR_DIM must be non-negative")
        self.temporal_lidar_data = {
            name: np.ones(self.temporal_lidar_dim) * 10 for name in self.agent_names
        }
        self.record_raw_lidar = _env_bool("DRL_MULTI_RECORD_RAW_LIDAR", False)
        self.raw_lidar_voxel_size = _env_float(
            "DRL_MULTI_RAW_LIDAR_VOXEL_SIZE", 0.05
        )
        self.raw_lidar_max_range = _env_float("DRL_MULTI_RAW_LIDAR_MAX_RANGE", 6.0)
        if self.raw_lidar_voxel_size <= 0.0:
            raise ValueError("DRL_MULTI_RAW_LIDAR_VOXEL_SIZE must be positive")
        if self.raw_lidar_max_range <= 0.0:
            raise ValueError("DRL_MULTI_RAW_LIDAR_MAX_RANGE must be positive")
        self.raw_lidar_points = {
            name: np.empty((0, 3), dtype=np.float32) for name in self.agent_names
        }
        self.velodyne_update_counts = {name: 0 for name in self.agent_names}
        self.odom_update_counts = {name: 0 for name in self.agent_names}
        self.velodyne_timestamps = {
            name: float("-inf") for name in self.agent_names
        }
        self.odom_timestamps = {name: float("-inf") for name in self.agent_names}
        self.sim_clock_time = float("-inf")
        self.sim_clock_update_count = 0
        self.fixed_agent_interfaces_ready = False
        self.last_odom = {name: None for name in self.agent_names}
        self.previous_distances = {name: None for name in self.agent_names}
        self.previous_nearest_robot_distances = {
            name: None for name in self.agent_names
        }
        self.yield_wait_steps = {name: 0 for name in self.agent_names}
        self.yield_nearest_distances = {name: None for name in self.agent_names}
        self.goal_positions = {name: np.array([1.0, 0.0]) for name in self.agent_names}
        self.robot_positions = {name: np.array([0.0, 0.0]) for name in self.agent_names}
        self.set_self_states = {name: self._create_model_state(name) for name in self.agent_names}
        self.last_step_info = self._empty_last_step_info()
        self.last_reward_neighbors = {name: [] for name in self.agent_names}
        self.last_interaction_rewards = {name: 0.0 for name in self.agent_names}
        self.last_active_visible_neighbor_counts = {
            name: 0 for name in self.agent_names
        }
        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.environment_dim]]
        for m in range(self.environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / self.environment_dim]
            )
        self.gaps[-1][-1] += 0.03
        self.temporal_lidar_gaps = None
        if self.temporal_lidar_dim:
            self.temporal_lidar_gaps = build_front_lidar_gaps(
                self.temporal_lidar_dim
            )

        port = os.environ.get("ROS_PORT_SIM", "11311")
        ros_master_uri = os.environ.get("ROS_MASTER_URI", "").strip()
        should_launch_roscore = not ros_master_uri
        if ros_master_uri:
            try:
                host_port = ros_master_uri.split("://", 1)[-1].rstrip("/")
                host, uri_port = host_port.split(":", 1)
                with socket.create_connection((host, int(uri_port)), timeout=1.0):
                    should_launch_roscore = False
            except Exception:
                should_launch_roscore = True
        if should_launch_roscore:
            self.roscore_process = subprocess.Popen(["roscore", "-p", port])
            print("Roscore launched!")
        else:
            print("Using existing ROS master:", ros_master_uri)

        rospy.init_node("multi_agent_gym", anonymous=True)
        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", launchfile)
        if not path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")

        roslaunch_command = ["roslaunch", "-p", port, fullpath]
        if self.fixed_physics_step_size is not None:
            roslaunch_command.append("paused:=true")
        self.roslaunch_process = subprocess.Popen(roslaunch_command)
        print("Gazebo launched!")

        self.vel_pubs = {
            name: rospy.Publisher(f"/{name}/cmd_vel", Twist, queue_size=1)
            for name in self.agent_names
        }
        self.set_state = rospy.Publisher("gazebo/set_model_state", ModelState, queue_size=20)
        self.set_model_state_service = rospy.ServiceProxy(
            "/gazebo/set_model_state", SetModelState
        )
        self.get_world_properties_service = rospy.ServiceProxy(
            "/gazebo/get_world_properties", GetWorldProperties
        )
        self.get_physics_properties_service = rospy.ServiceProxy(
            "/gazebo/get_physics_properties", GetPhysicsProperties
        )
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.reset_simulation_proxy = rospy.ServiceProxy(
            "/gazebo/reset_simulation", Empty
        )
        self.goal_publisher = rospy.Publisher("goal_points", MarkerArray, queue_size=10)
        self.clock_subscriber = rospy.Subscriber(
            "/clock", Clock, self._clock_callback, queue_size=1
        )

        self.velodyne_subscribers = []
        self.odom_subscribers = []
        for name in self.agent_names:
            self.velodyne_subscribers.append(
                rospy.Subscriber(
                    f"/{name}/velodyne_points",
                    PointCloud2,
                    self._make_velodyne_callback(name),
                    queue_size=1,
                )
            )
            self.odom_subscribers.append(
                rospy.Subscriber(
                    f"/{name}/odom",
                    Odometry,
                    self._make_odom_callback(name),
                    queue_size=1,
                )
            )

    @staticmethod
    def _create_model_state(model_name):
        state = ModelState()
        state.model_name = model_name
        state.pose.position.x = 0.0
        state.pose.position.y = 0.0
        state.pose.position.z = 0.0
        state.pose.orientation.x = 0.0
        state.pose.orientation.y = 0.0
        state.pose.orientation.z = 0.0
        state.pose.orientation.w = 1.0
        return state

    def _set_model_state(self, state, timeout=10.0):
        if self.fixed_physics_step_size is None:
            self.set_state.publish(state)
            return
        rospy.wait_for_service("/gazebo/set_model_state")
        deadline = time.monotonic() + timeout
        last_status = ""
        while time.monotonic() < deadline:
            try:
                response = self.set_model_state_service(state)
                if response.success:
                    return
                last_status = response.status_message
            except rospy.ServiceException as exc:
                last_status = str(exc)
            time.sleep(0.05)
        raise RuntimeError(
            "Could not set Gazebo model state for %s: %s"
            % (state.model_name, last_status)
        )

    def close(self):
        for process in (self.roslaunch_process, self.roscore_process):
            if process is None or process.poll() is not None:
                continue
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def _load_curriculum_cases(self):
        cases_path = os.environ.get("DRL_MULTI_CURRICULUM_CASES", "").strip()
        if not cases_path:
            raise ValueError(
                "DRL_MULTI_CURRICULUM_CASES must point to a JSON case file when "
                "DRL_MULTI_SCENARIO=curriculum"
            )
        if not os.path.isabs(cases_path):
            cases_path = os.path.join(os.getcwd(), cases_path)
        with open(cases_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        cases = []
        if isinstance(payload, dict):
            base_dir = os.path.dirname(cases_path)
            for source in payload.get("case_files", []):
                source = {"path": source} if isinstance(source, str) else source
                source_path = source.get("path", "").strip()
                if not source_path:
                    raise ValueError("Each curriculum case_files entry needs a path")
                if not os.path.isabs(source_path):
                    source_path = os.path.join(base_dir, source_path)
                with open(source_path, "r", encoding="utf-8") as source_fh:
                    source_payload = json.load(source_fh)
                source_cases = (
                    source_payload.get("cases", source_payload)
                    if isinstance(source_payload, dict)
                    else source_payload
                )
                if not isinstance(source_cases, list):
                    raise ValueError(f"Curriculum source must contain cases: {source_path}")
                weight_scale = float(source.get("weight_scale", 1.0))
                for source_case in source_cases:
                    case = dict(source_case)
                    if source.get("group"):
                        case["group"] = source["group"]
                    case["weight"] = float(case.get("weight", 1.0)) * weight_scale
                    cases.append(case)
            cases.extend(payload.get("cases", []))
        else:
            cases = payload
        if not isinstance(cases, list) or not cases:
            raise ValueError("Curriculum case file must contain a non-empty case list")
        normalized = []
        for idx, case in enumerate(cases):
            layout = str(case.get("layout", "fixed")).strip().lower()
            if layout == "standard":
                normalized.append(case)
                continue
            if layout != "fixed":
                raise ValueError(
                    f"Curriculum case {idx} layout must be fixed or standard"
                )
            if "agents" not in case or not isinstance(case["agents"], dict):
                raise ValueError(f"Curriculum case {idx} must define an agents object")
            missing = [name for name in self.agent_names if name not in case["agents"]]
            if missing:
                raise ValueError(
                    f"Curriculum case {idx} is missing agents: {', '.join(missing)}"
                )
            for name in self.agent_names:
                for key in ("start", "goal"):
                    pos = case["agents"][name].get(key)
                    if not isinstance(pos, list) or len(pos) != 2:
                        raise ValueError(
                            f"Curriculum case {idx} agent {name} {key} must be [x, y]"
                        )
                    if not check_pos(float(pos[0]), float(pos[1])):
                        raise ValueError(
                            f"Curriculum case {idx} agent {name} {key}={pos} is invalid in the map"
                        )
            normalized.append(case)
        return normalized

    def _load_manifest_cases(self):
        manifest_path = os.environ.get("DRL_MULTI_MANIFEST_PATH", "").strip()
        if not manifest_path:
            raise ValueError(
                "DRL_MULTI_MANIFEST_PATH must point to a split JSON file when "
                "DRL_MULTI_SCENARIO=manifest"
            )
        if not os.path.isabs(manifest_path):
            manifest_path = os.path.join(os.getcwd(), manifest_path)
        payload = load_manifest_dataset(manifest_path)
        self.manifest_path = os.path.abspath(manifest_path)
        self.manifest_dataset_id = str(payload.get("dataset_id", "unknown"))
        return validate_manifest_scenarios(payload["scenarios"], self.agent_names)

    def set_manifest_path(self, manifest_path):
        """Switch fixed cases on the existing Gazebo instance without respawning it."""
        if self.scenario_mode != "manifest":
            raise ValueError("set_manifest_path requires scenario_mode=manifest")
        manifest_path = os.path.abspath(os.path.expanduser(str(manifest_path)))
        previous_path = os.environ.get("DRL_MULTI_MANIFEST_PATH")
        os.environ["DRL_MULTI_MANIFEST_PATH"] = manifest_path
        try:
            cases = self._load_manifest_cases()
        finally:
            if previous_path is None:
                os.environ.pop("DRL_MULTI_MANIFEST_PATH", None)
            else:
                os.environ["DRL_MULTI_MANIFEST_PATH"] = previous_path
        self.curriculum_cases = cases
        self.curriculum_case_index = 0
        self._reset_manifest_band_sampling()
        self.current_curriculum_case = None
        sampling = os.environ.get("DRL_MULTI_MANIFEST_SAMPLING", "cycle")
        print(
            "Manifest switched: %s | scenarios=%i | sampling=%s | path=%s"
            % (
                self.manifest_dataset_id,
                len(cases),
                sampling,
                self.manifest_path,
            )
        )

    def _current_case_uses_standard_layout(self):
        return bool(
            self.scenario_mode == "curriculum"
            and self.current_curriculum_case
            and str(self.current_curriculum_case.get("layout", "fixed")).lower()
            == "standard"
        )

    def _select_curriculum_case(self):
        variable = (
            "DRL_MULTI_MANIFEST_SAMPLING"
            if self.scenario_mode == "manifest"
            else "DRL_MULTI_CURRICULUM_SAMPLING"
        )
        mode = os.environ.get(variable, "cycle").strip().lower()
        if mode == "random":
            weights = [
                max(float(case.get("weight", 1.0)), 0.0)
                for case in self.curriculum_cases
            ]
            if sum(weights) <= 0:
                return random.choice(self.curriculum_cases)
            return random.choices(self.curriculum_cases, weights=weights, k=1)[0]
        if mode == "balanced_cycle":
            schedule = ("deep", "deep", "close", "close", "margin")
            if not self.manifest_band_cases:
                raise ValueError(
                    "balanced_cycle requires manifest scenarios with view.interaction_band"
                )
            # Interleave the bands while all are available. Once a band is
            # exhausted, skip it rather than duplicating cases; this preserves
            # complete finite-pool coverage without silently oversampling easy
            # or hard strata. A new pass starts after every case is consumed.
            for offset in range(len(schedule)):
                slot = (self.manifest_band_schedule_index + offset) % len(schedule)
                band = schedule[slot]
                cases = self.manifest_band_cases.get(band, ())
                if self.manifest_band_indices.get(band, 0) >= len(cases):
                    continue
                index = self.manifest_band_indices[band]
                self.manifest_band_indices[band] += 1
                self.manifest_band_schedule_index = (slot + 1) % len(schedule)
                return cases[index]
            self._reset_manifest_band_sampling()
            return self._select_curriculum_case()
        if mode not in ("cycle", "paired_cycle"):
            raise ValueError(
                f"{variable} must be cycle, paired_cycle, random, or balanced_cycle"
            )
        case_index = self.curriculum_case_index
        if mode == "paired_cycle":
            case_index //= 2
        case = self.curriculum_cases[case_index % len(self.curriculum_cases)]
        self.curriculum_case_index += 1
        return case

    def _reset_manifest_band_sampling(self):
        self.manifest_band_cases = {}
        self.manifest_band_indices = {}
        self.manifest_band_schedule_index = 0
        for case in self.curriculum_cases:
            band = case.get("view", {}).get("interaction_band")
            if band in ("deep", "close", "margin"):
                self.manifest_band_cases.setdefault(band, []).append(case)
                self.manifest_band_indices.setdefault(band, 0)

    def manifest_sampling_state(self):
        return {
            "curriculum_case_index": int(self.curriculum_case_index),
            "band_indices": {
                str(band): int(index)
                for band, index in self.manifest_band_indices.items()
            },
            "band_schedule_index": int(self.manifest_band_schedule_index),
        }

    def restore_manifest_sampling_state(self, state):
        if not state:
            return
        self.curriculum_case_index = int(state.get("curriculum_case_index", 0))
        saved_band_indices = state.get("band_indices") or {}
        for band in self.manifest_band_indices:
            index = int(saved_band_indices.get(band, 0))
            self.manifest_band_indices[band] = min(
                max(index, 0), len(self.manifest_band_cases.get(band, ()))
            )
        self.manifest_band_schedule_index = int(
            state.get("band_schedule_index", 0)
        ) % 5

    def _curriculum_agent_position(self, name, key):
        value = self.current_curriculum_case["agents"][name][key]
        if len(value) != 2:
            raise ValueError(f"{name}.{key} must contain [x, y]")
        base = np.array([float(value[0]), float(value[1])])
        if self.scenario_mode == "manifest":
            return base
        jitter_key = f"{key}_jitter"
        jitter = self.current_curriculum_case["agents"][name].get(jitter_key)
        if jitter is None:
            return base
        if not isinstance(jitter, list) or len(jitter) != 2:
            raise ValueError(f"{name}.{jitter_key} must contain [dx, dy]")
        dx = float(jitter[0])
        dy = float(jitter[1])
        for _ in range(20):
            candidate = np.array(
                [
                    base[0] + np.random.uniform(-dx, dx),
                    base[1] + np.random.uniform(-dy, dy),
                ]
            )
            if check_pos(float(candidate[0]), float(candidate[1])):
                return candidate
        return base

    def _apply_curriculum_boxes(self):
        if self._current_case_uses_standard_layout():
            self.random_box()
            return
        boxes = self.current_curriculum_case.get("boxes")
        if boxes is None:
            self.random_box()
            return
        for idx in range(4):
            box_state = ModelState()
            box_state.model_name = "cardboard_box_" + str(idx)
            if idx < len(boxes):
                box = boxes[idx]
                if len(box) != 2:
                    raise ValueError("Each curriculum box must contain [x, y]")
                box_state.pose.position.x = float(box[0])
                box_state.pose.position.y = float(box[1])
            else:
                box_state.pose.position.x = 20.0 + idx
                box_state.pose.position.y = 20.0
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self._set_model_state(box_state)

    def _make_velodyne_callback(self, name):
        def callback(msg):
            data = list(pc2.read_points(msg, skip_nans=False, field_names=("x", "y", "z")))
            agent_scan = np.ones(self.environment_dim) * 10
            temporal_scan = np.ones(self.temporal_lidar_dim) * 10
            raw_front_points = []
            for point in data:
                if point[2] <= -0.2:
                    continue
                dot = point[0]
                mag1 = math.sqrt(point[0] ** 2 + point[1] ** 2)
                if mag1 == 0:
                    continue
                beta = math.acos(dot / mag1) * np.sign(point[1])
                dist = math.sqrt(point[0] ** 2 + point[1] ** 2 + point[2] ** 2)

                if (
                    self.record_raw_lidar
                    and self.gaps[0][0] <= beta < self.gaps[-1][1]
                    and mag1 <= self.raw_lidar_max_range
                ):
                    raw_front_points.append((point[0], point[1], point[2]))

                for idx, gap in enumerate(self.gaps):
                    if gap[0] <= beta < gap[1]:
                        agent_scan[idx] = min(agent_scan[idx], dist)
                        break
                if self.temporal_lidar_dim:
                    temporal_index = int(
                        np.searchsorted(
                            self.temporal_lidar_gaps[:, 1], beta, side="right"
                        )
                    )
                    if (
                        0 <= temporal_index < self.temporal_lidar_dim
                        and self.temporal_lidar_gaps[temporal_index, 0]
                        <= beta
                        < self.temporal_lidar_gaps[temporal_index, 1]
                    ):
                        temporal_scan[temporal_index] = min(
                            temporal_scan[temporal_index], dist
                        )
            self.velodyne_data[name] = agent_scan
            if self.temporal_lidar_dim:
                self.temporal_lidar_data[name] = temporal_scan
            if self.record_raw_lidar:
                if raw_front_points:
                    points = np.asarray(raw_front_points, dtype=np.float32)
                    voxels = np.floor(
                        points / self.raw_lidar_voxel_size
                    ).astype(np.int32)
                    _, indices = np.unique(voxels, axis=0, return_index=True)
                    self.raw_lidar_points[name] = points[np.sort(indices)]
                else:
                    self.raw_lidar_points[name] = np.empty((0, 3), dtype=np.float32)
            self.velodyne_timestamps[name] = float(msg.header.stamp.to_sec())
            self.velodyne_update_counts[name] += 1

        return callback

    def _make_odom_callback(self, name):
        def callback(msg):
            self.last_odom[name] = msg
            self.odom_timestamps[name] = float(msg.header.stamp.to_sec())
            self.odom_update_counts[name] += 1

        return callback

    def _clock_callback(self, msg):
        self.sim_clock_time = float(msg.clock.to_sec())
        self.sim_clock_update_count += 1

    def _empty_last_step_info(self):
        return {
            "agents": {
                name: {
                    "target": False,
                    "collision": False,
                    "distance": None,
                    "progress": 0.0,
                    "min_laser": None,
                    "nearest_robot_distance": None,
                    "reward": 0.0,
                    "raw_reward": 0.0,
                    "interaction_reward": 0.0,
                    "anti_stagnation_reward": 0.0,
                    "wall_clearance_reward": 0.0,
                    "local_navigation_reward": 0.0,
                    "robot_proximity_reward": 0.0,
                    "robot_clearance_reward": 0.0,
                    "yield_priority_reward": 0.0,
                    "reward_neighbors": [],
                    "active_visible_neighbor_count": 0,
                    "nearest_active_visible_neighbor_distance": None,
                }
                for name in self.agent_names
            },
            "mean_reward": 0.0,
            "success_count": 0,
            "collision_count": 0,
            "active_neighbor_agent_count": 0,
            "mean_active_visible_neighbors": 0.0,
            "max_active_visible_neighbors": 0,
            "mean_interaction_reward": 0.0,
            "mean_anti_stagnation_reward": 0.0,
            "mean_yield_priority_reward": 0.0,
        }

    def set_cooperative_reward(self, enabled):
        self.cooperative_reward = enabled

    def set_anti_stagnation_reward(self, enabled):
        self.anti_stagnation_reward = enabled

    def set_wall_clearance_reward(self, enabled):
        self.wall_clearance_reward = enabled

    def set_local_navigation_reward(self, enabled):
        self.local_navigation_reward = enabled

    def wait_for_odom(self, name, timeout=60.0, recover_with_unpause=True):
        if self.last_odom[name] is not None:
            return
        if recover_with_unpause:
            try:
                rospy.wait_for_service("/gazebo/unpause_physics", timeout=5.0)
                self.unpause()
                time.sleep(max(TIME_DELTA, 0.2))
            except (rospy.ROSException, rospy.ServiceException):
                pass
        try:
            self.last_odom[name] = rospy.wait_for_message(
                f"/{name}/odom", Odometry, timeout=timeout
            )
            return
        except rospy.ROSException as exc:
            if not recover_with_unpause:
                raise TimeoutError(
                    f"Timed out waiting for /{name}/odom after {timeout} seconds"
                ) from exc
            try:
                rospy.wait_for_service("/gazebo/unpause_physics", timeout=5.0)
                self.unpause()
                time.sleep(max(TIME_DELTA * 2, 0.4))
                self.last_odom[name] = rospy.wait_for_message(
                    f"/{name}/odom", Odometry, timeout=timeout
                )
                return
            except (rospy.ROSException, rospy.ServiceException) as retry_exc:
                raise TimeoutError(
                    f"Timed out waiting for /{name}/odom after reset recovery"
                ) from retry_exc

    def wait_for_all_odom(self, timeout_per_agent=2.0, attempts=5):
        missing = []
        for attempt in range(attempts):
            missing = []
            for name in self.agent_names:
                if self.last_odom[name] is not None:
                    continue
                try:
                    self.wait_for_odom(
                        name,
                        timeout=timeout_per_agent,
                        recover_with_unpause=False,
                    )
                except TimeoutError:
                    missing.append(name)
            if not missing:
                return
            try:
                rospy.wait_for_service("/gazebo/unpause_physics", timeout=5.0)
                self.unpause()
            except (rospy.ROSException, rospy.ServiceException):
                pass
            time.sleep(TIME_DELTA)
        raise TimeoutError(
            "Timed out waiting for odom topics after reset: "
            + ", ".join(f"/{name}/odom" for name in missing)
        )

    def _get_robot_yaw(self, name):
        odom = self.last_odom[name]
        quaternion = Quaternion(
            odom.pose.pose.orientation.w,
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        return round(euler[2], 4)

    def _build_state(self, name, action):
        self.wait_for_odom(name)
        odom = self.last_odom[name]
        self.robot_positions[name] = np.array(
            [odom.pose.pose.position.x, odom.pose.pose.position.y]
        )
        angle = self._get_robot_yaw(name)

        goal_x, goal_y = self.goal_positions[name]
        distance = np.linalg.norm(self.robot_positions[name] - self.goal_positions[name])

        skew_x = goal_x - self.robot_positions[name][0]
        skew_y = goal_y - self.robot_positions[name][1]
        dot = skew_x
        mag1 = math.sqrt(skew_x ** 2 + skew_y ** 2)
        mag2 = 1.0
        cos_beta = np.clip(dot / (mag1 * mag2), -1.0, 1.0) if mag1 > 0 else 1.0
        beta = math.acos(cos_beta)
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = -beta
        theta = beta - angle

        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        robot_state = [distance, theta, action[0], action[1]]
        return np.append([self.velodyne_data[name]], robot_state), distance

    def _sync_robot_positions_from_odom(self):
        for name in self.agent_names:
            odom = self.last_odom[name]
            self.robot_positions[name] = np.array(
                [odom.pose.pose.position.x, odom.pose.pose.position.y]
            )

    def _compute_visible_neighbors(self, name, active_names=None):
        neighbors = []
        origin = self.robot_positions[name]
        yaw = self._get_robot_yaw(name)
        heading = np.array([math.cos(yaw), math.sin(yaw)])

        for other_name in self.agent_names:
            if other_name == name:
                continue
            if active_names is not None and other_name not in active_names:
                continue
            offset = self.robot_positions[other_name] - origin
            distance = np.linalg.norm(offset)
            if distance == 0 or distance > self.reward_neighbor_radius:
                continue

            direction = offset / distance
            dot = np.clip(np.dot(heading, direction), -1.0, 1.0)
            relative_angle = math.acos(dot)
            if relative_angle <= self.reward_neighbor_fov:
                neighbors.append(other_name)
        neighbors.sort(
            key=lambda other: np.linalg.norm(self.robot_positions[other] - origin)
        )
        return neighbors

    def build_neighbor_context(
        self, actions, max_neighbors=9, include_actions=True, active_mask=None
    ):
        contexts = []
        feature_dim = context_feature_dim(
            self.neighbor_context_mode, legacy_include_actions=include_actions
        )
        active_names = None
        if self.active_neighbors_only and active_mask is not None:
            active_names = {
                name
                for idx, name in enumerate(self.agent_names)
                if idx < len(active_mask) and active_mask[idx]
            }
        action_by_name = {
            name: np.array(actions[idx], dtype=np.float32)
            for idx, name in enumerate(self.agent_names)
        }

        for name in self.agent_names:
            if active_names is not None and name not in active_names:
                contexts.append(np.zeros(max_neighbors * feature_dim, dtype=np.float32))
                continue
            origin = self.robot_positions[name]
            yaw = self._get_robot_yaw(name)
            context = []
            for other_name in self._compute_visible_neighbors(
                name, active_names=active_names
            )[:max_neighbors]:
                offset = self.robot_positions[other_name] - origin
                distance = float(np.linalg.norm(offset))
                bearing = math.atan2(offset[1], offset[0]) - yaw
                while bearing > np.pi:
                    bearing -= 2 * np.pi
                while bearing < -np.pi:
                    bearing += 2 * np.pi
                if self.neighbor_context_mode == "ego_motion":
                    neighbor_features = ego_motion_features(
                        offset,
                        yaw,
                        self._get_robot_yaw(other_name),
                        action_by_name.get(name, np.zeros(2)),
                        action_by_name.get(other_name, np.zeros(2)),
                    )
                else:
                    relative_position = offset
                    if self.neighbor_context_mode == "ego_geometry":
                        relative_position = world_to_ego(offset, yaw)
                    neighbor_features = [
                        float(relative_position[0]),
                        float(relative_position[1]),
                        distance,
                        float(bearing),
                    ]
                if self.neighbor_context_mode == "legacy" and include_actions:
                    other_action = action_by_name.get(other_name, np.zeros(2))
                    neighbor_features.extend(
                        [float(other_action[0]), float(other_action[1])]
                    )
                if self.neighbor_context_mode != "ego_motion":
                    neighbor_features.append(1.0)
                context.extend(neighbor_features)

            missing = max_neighbors - len(context) // feature_dim
            if missing > 0:
                context.extend([0.0] * missing * feature_dim)
            contexts.append(np.array(context, dtype=np.float32))
        return contexts

    def _apply_cooperative_reward(self, rewards, active_mask, step_agents_info=None):
        adjusted = rewards.copy()
        self.last_reward_neighbors = {name: [] for name in self.agent_names}
        self.last_interaction_rewards = {name: 0.0 for name in self.agent_names}
        self.last_active_visible_neighbor_counts = {
            name: 0 for name in self.agent_names
        }
        active_names = None
        if self.active_neighbors_only:
            active_names = {
                name
                for idx, name in enumerate(self.agent_names)
                if idx < len(active_mask) and active_mask[idx]
            }
        for idx, name in enumerate(self.agent_names):
            if not active_mask[idx]:
                continue
            visible_neighbors = self._compute_visible_neighbors(
                name, active_names=active_names
            )
            visible = [name] + visible_neighbors
            self.last_reward_neighbors[name] = [n for n in visible if n != name]
            self.last_active_visible_neighbor_counts[name] = len(
                self.last_reward_neighbors[name]
            )
            if self.cooperative_reward_mode == "interaction_only":
                interaction_reward = self._compute_interaction_reward(
                    name,
                    self.last_reward_neighbors[name],
                    step_agents_info[name]["progress"] if step_agents_info else 0.0,
                )
                self.last_interaction_rewards[name] = interaction_reward
                adjusted[idx] = float(rewards[idx] + interaction_reward)
                continue
            if (
                self.cooperative_reward_self_weight is not None
                and self.last_reward_neighbors[name]
            ):
                self_weight = float(self.cooperative_reward_self_weight)
                neighbor_rewards = np.array(
                    [
                        rewards[self.agent_names.index(n)]
                        for n in self.last_reward_neighbors[name]
                    ],
                    dtype=np.float32,
                )
                if self.cooperative_reward_distance_weighted:
                    distances = np.array(
                        [
                            np.linalg.norm(
                                self.robot_positions[n] - self.robot_positions[name]
                            )
                            for n in self.last_reward_neighbors[name]
                        ],
                        dtype=np.float32,
                    )
                    sigma = max(float(self.cooperative_reward_sigma), 1e-6)
                    weights = np.exp(-distances / sigma)
                    neighbor_reward = float(np.sum(weights * neighbor_rewards) / np.sum(weights))
                else:
                    neighbor_reward = float(np.mean(neighbor_rewards))
                adjusted[idx] = float(
                    self_weight * rewards[idx]
                    + (1.0 - self_weight) * neighbor_reward
                )
            else:
                adjusted[idx] = float(
                    np.mean([rewards[self.agent_names.index(n)] for n in visible])
                )
            if self.cooperative_reward_mode == "average_plus_interaction":
                interaction_reward = self._compute_interaction_reward(
                    name,
                    self.last_reward_neighbors[name],
                    step_agents_info[name]["progress"] if step_agents_info else 0.0,
                )
                self.last_interaction_rewards[name] = interaction_reward
                adjusted[idx] = float(adjusted[idx] + interaction_reward)
        return adjusted

    def _compute_interaction_reward(self, name, visible_neighbors, progress):
        if not visible_neighbors:
            return 0.0

        distances = np.array(
            [
                np.linalg.norm(self.robot_positions[other_name] - self.robot_positions[name])
                for other_name in visible_neighbors
            ],
            dtype=np.float32,
        )
        safe_distance = max(float(self.interaction_safe_distance), 1e-6)
        close_pressure = np.maximum(0.0, safe_distance - distances) / safe_distance
        close_penalty = float(np.mean(close_pressure))

        stagnation_penalty = 0.0
        if progress < 0.002:
            stagnation_penalty = float(self.interaction_stagnation_penalty)

        return -float(self.interaction_close_penalty) * close_penalty - stagnation_penalty

    def _compute_anti_stagnation_penalty(
        self,
        target,
        collision,
        action,
        min_laser,
        progress,
        nearest_robot_distance=float("inf"),
    ):
        if not self.anti_stagnation_reward or target or collision:
            return 0.0
        if (
            self.robot_safe_distance > 0.0
            and np.isfinite(nearest_robot_distance)
            and nearest_robot_distance < self.robot_safe_distance
        ):
            return 0.0
        if min_laser is None or min_laser < self.anti_stagnation_min_laser:
            return 0.0
        if (
            action[0] < self.anti_stagnation_linear_threshold
            and abs(progress) < self.anti_stagnation_progress_threshold
        ):
            return float(self.anti_stagnation_penalty)
        return 0.0

    def _compute_safe_recovery_reward(
        self,
        target,
        collision,
        action,
        min_laser,
        progress,
        nearest_robot_distance=float("inf"),
    ):
        if not self.safe_recovery_reward or target or collision:
            return 0.0
        if min_laser is None or min_laser < self.safe_recovery_min_laser:
            return 0.0
        if (
            self.safe_recovery_robot_distance > 0.0
            and np.isfinite(nearest_robot_distance)
            and nearest_robot_distance < self.safe_recovery_robot_distance
        ):
            return 0.0
        linear = max(float(action[0]), 0.0)
        reward = 0.0
        if (
            linear < self.safe_recovery_linear_threshold
            and progress < self.safe_recovery_progress_threshold
        ):
            reward -= float(self.safe_recovery_penalty)
            reward -= (
                self.safe_recovery_idle_penalty_weight
                * max(self.safe_recovery_linear_threshold - linear, 0.0)
            )
        elif progress > self.safe_recovery_progress_threshold:
            reward += (
                self.safe_recovery_progress_bonus_weight
                * min(progress / max(self.safe_recovery_progress_threshold, 1e-6), 3.0)
                / 3.0
            )
        return float(reward)

    def _compute_safe_recovery_penalty(
        self,
        target,
        collision,
        action,
        min_laser,
        progress,
        nearest_robot_distance=float("inf"),
    ):
        return max(
            -self._compute_safe_recovery_reward(
                target,
                collision,
                action,
                min_laser,
                progress,
                nearest_robot_distance,
            ),
            0.0,
        )

    def _compute_wall_clearance_penalty(self, target, collision, action, min_laser):
        if not self.wall_clearance_reward or target or collision:
            return 0.0
        if min_laser is None:
            return 0.0
        safe_distance = max(float(self.wall_clearance_safe_distance), COLLISION_DIST)
        if min_laser >= safe_distance:
            return 0.0
        pressure = (safe_distance - min_laser) / safe_distance
        motion_scale = (
            1.0
            + self.wall_clearance_speed_weight * max(float(action[0]), 0.0)
            + self.wall_clearance_turn_weight * abs(float(action[1]))
        )
        return float(self.wall_clearance_penalty) * pressure * motion_scale

    def _compute_local_navigation_bonus(
        self, target, collision, action, distance, theta, min_laser
    ):
        if not self.local_navigation_reward or target or collision:
            return 0.0
        if self.scenario_mode in ("curriculum", "manifest") and self.current_curriculum_case:
            case_override = self.current_curriculum_case.get("local_navigation_reward")
            if case_override is not None and not bool(case_override):
                return 0.0
        if min_laser is not None and min_laser < COLLISION_DIST + 0.03:
            return 0.0

        linear = max(float(action[0]), 0.0)
        angular = abs(float(action[1]))
        heading_alignment = math.cos(float(theta))

        aligned_forward_bonus = (
            self.local_navigation_heading_weight
            * linear
            * max(heading_alignment, 0.0)
        )
        wrong_way_penalty = (
            self.local_navigation_wrong_way_penalty
            * linear
            * max(-heading_alignment, 0.0)
        )

        near_goal_turn_bonus = 0.0
        if (
            distance < self.local_navigation_near_goal_distance
            and abs(theta) > self.local_navigation_heading_error
        ):
            turn_need = min(abs(float(theta)) / math.pi, 1.0)
            low_speed_gate = max(0.0, 1.0 - linear)
            near_goal_turn_bonus = (
                self.local_navigation_turn_weight
                * turn_need
                * angular
                * low_speed_gate
            )

        return float(aligned_forward_bonus + near_goal_turn_bonus - wrong_way_penalty)

    def _compute_robot_proximity_penalty(self, action, nearest_robot_distance):
        safe_distance = self.robot_safe_distance
        if (
            safe_distance <= 0.0
            or not np.isfinite(nearest_robot_distance)
            or nearest_robot_distance >= safe_distance
        ):
            return 0.0

        clearance_deficit = safe_distance - nearest_robot_distance
        proximity_pressure = clearance_deficit / safe_distance
        linear_speed = max(float(action[0]), 0.0)
        return (
            self.robot_proximity_penalty_weight * clearance_deficit
            + self.robot_proximity_speed_penalty_weight
            * proximity_pressure
            * linear_speed
        )

    def _compute_robot_clearance_reward(
        self,
        target,
        collision,
        previous_robot_distance,
        nearest_robot_distance,
    ):
        if target or collision or self.robot_safe_distance <= 0.0:
            return 0.0
        if previous_robot_distance is None:
            return 0.0
        if not (
            np.isfinite(previous_robot_distance)
            and np.isfinite(nearest_robot_distance)
        ):
            return 0.0
        if (
            min(previous_robot_distance, nearest_robot_distance)
            >= self.robot_safe_distance
        ):
            return 0.0

        clearance_gain = nearest_robot_distance - previous_robot_distance
        capped_gain = np.clip(
            clearance_gain,
            -self.robot_clearance_reward_max_gain,
            self.robot_clearance_reward_max_gain,
        )
        return self.robot_clearance_reward_weight * capped_gain

    def _compute_yield_priority_reward(
        self,
        name,
        action,
        distance_to_goal,
        active_names,
        target=False,
        collision=False,
    ):
        if not self.yield_priority_reward or target or collision:
            return 0.0
        if self.robot_safe_distance <= 0.0:
            return 0.0

        linear = max(float(action[0]), 0.0)
        stop_gate = max(0.0, 1.0 - linear / max(self.yield_priority_stop_linear, 1e-6))
        yield_reward = 0.0
        origin = self.robot_positions[name]
        visible_neighbors = self._compute_visible_neighbors(
            name, active_names=active_names
        )
        nearest_conflict_distance = float("inf")
        should_yield = False

        for other_name in visible_neighbors:
            offset = self.robot_positions[other_name] - origin
            robot_distance = float(np.linalg.norm(offset))
            if robot_distance <= 0.0:
                continue
            nearest_conflict_distance = min(nearest_conflict_distance, robot_distance)

            if (
                self.emergency_stop_distance > 0.0
                and robot_distance < self.emergency_stop_distance
            ):
                pressure = (
                    self.emergency_stop_distance - robot_distance
                ) / self.emergency_stop_distance
                yield_reward -= self.emergency_stop_penalty_weight * pressure * linear
                yield_reward += self.emergency_stop_bonus_weight * pressure * stop_gate

            if robot_distance >= self.yield_priority_distance:
                continue

            other_goal_distance = float(
                np.linalg.norm(
                    self.robot_positions[other_name]
                    - self.goal_positions[other_name]
                )
            )
            goal_delta = float(distance_to_goal - other_goal_distance)
            if goal_delta <= self.yield_priority_goal_margin:
                continue

            should_yield = True
            pressure = (
                self.yield_priority_distance - robot_distance
            ) / self.yield_priority_distance
            priority = min(
                goal_delta / max(self.yield_priority_distance, 1e-6),
                1.0,
            )
            yield_reward -= (
                self.yield_priority_penalty_weight * pressure * priority * linear
            )
            yield_reward += (
                self.yield_priority_bonus_weight * pressure * priority * stop_gate
            )

        wait_steps = getattr(self, "yield_wait_steps", {}).get(name, 0)
        previous_conflict_distance = getattr(
            self, "yield_nearest_distances", {}
        ).get(name)

        if should_yield:
            wait_steps += 1 if linear <= self.yield_priority_stop_linear else 0
            if (
                linear <= self.yield_priority_stop_linear
                and previous_conflict_distance is not None
                and np.isfinite(previous_conflict_distance)
                and np.isfinite(nearest_conflict_distance)
            ):
                clearance_gain = nearest_conflict_distance - previous_conflict_distance
                if clearance_gain > 0.0:
                    capped_gain = min(clearance_gain, self.robot_clearance_reward_max_gain)
                    yield_reward += (
                        self.yield_priority_clearance_bonus_weight * capped_gain
                    )
            if (
                self.yield_priority_max_wait_steps > 0
                and wait_steps > self.yield_priority_max_wait_steps
                and linear <= self.yield_priority_stop_linear
            ):
                stale = wait_steps - self.yield_priority_max_wait_steps
                yield_reward -= (
                    self.yield_priority_stale_wait_penalty_weight
                    * min(float(stale), 5.0)
                    * stop_gate
                )
        else:
            if wait_steps > 0:
                restart_gate = min(
                    linear / max(self.yield_priority_stop_linear, 1e-6), 1.0
                )
                yield_reward += (
                    self.yield_priority_restart_bonus_weight
                    * restart_gate
                    * min(wait_steps / max(self.yield_priority_max_wait_steps, 1), 1.0)
                )
            if (
                visible_neighbors
                and linear <= self.yield_priority_stop_linear
                and nearest_conflict_distance >= self.yield_priority_distance
            ):
                yield_reward -= self.yield_priority_stale_wait_penalty_weight * stop_gate
            wait_steps = 0

        if hasattr(self, "yield_wait_steps"):
            self.yield_wait_steps[name] = wait_steps
        if hasattr(self, "yield_nearest_distances"):
            self.yield_nearest_distances[name] = (
                nearest_conflict_distance
                if np.isfinite(nearest_conflict_distance)
                else None
            )
        return float(yield_reward)

    def _nearest_robot_distance(self, name):
        origin = self.robot_positions[name]
        distances = []
        for other_name in self.agent_names:
            if other_name == name:
                continue
            distances.append(np.linalg.norm(self.robot_positions[other_name] - origin))
        if not distances:
            return float("inf")
        return float(min(distances))

    def _nearest_visible_robot_distance(self, name, active_names):
        visible_neighbors = self._compute_visible_neighbors(
            name, active_names=active_names
        )
        if not visible_neighbors:
            return float("inf")
        origin = self.robot_positions[name]
        return float(
            min(
                np.linalg.norm(self.robot_positions[other_name] - origin)
                for other_name in visible_neighbors
            )
        )

    def _uses_capacity_layout(self):
        return (
            (
                self.scenario_mode == "standard"
                or self._current_case_uses_standard_layout()
            )
            and self.weak_coupling_layout
            and self.num_agents >= 5
        )

    def _sample_position(self, x_range, y_range):
        for _ in range(500):
            candidate = np.array(
                [np.random.uniform(*x_range), np.random.uniform(*y_range)]
            )
            if check_pos(candidate[0], candidate[1]):
                return candidate
        raise RuntimeError(
            "Could not sample a valid position in range "
            f"x={x_range}, y={y_range}. Check map capacity and obstacles."
        )

    def _agent_side_ranges(self, name):
        if self.scenario_mode == "dense":
            return self.dense_start_x_range, self.dense_start_y_range
        if not self.weak_coupling_layout:
            return (-4.5, 4.5), (-4.5, 4.5)
        if self.num_agents > 2:
            return (-4.5, 4.5), (-4.5, 4.5)
        if name == self.agent_names[0]:
            return (-4.2, -1.0), (-4.2, 4.2)
        return (1.0, 4.2), (-4.2, 4.2)

    def _sample_goal_candidate_for_agent(self, name):
        if self.scenario_mode == "dense":
            x_range, y_range = self._agent_side_ranges(name)
            while True:
                x_offset = random.uniform(*self.dense_goal_x_offset)
                y_offset = random.uniform(*self.dense_goal_y_offset)
                candidate = np.array(
                    [
                        np.clip(
                            self.robot_positions[name][0] + x_offset,
                            x_range[0],
                            x_range[1],
                        ),
                        np.clip(
                            self.robot_positions[name][1] + y_offset,
                            y_range[0],
                            y_range[1],
                        ),
                    ]
                )
                if (
                    np.linalg.norm(candidate - self.robot_positions[name])
                    < self.dense_goal_min_distance
                ):
                    continue
                return candidate

        if not self.weak_coupling_layout:
            return np.array(
                [
                    self.robot_positions[name][0] + random.uniform(self.lower, self.upper),
                    self.robot_positions[name][1] + random.uniform(self.lower, self.upper),
                ]
            )

        x_range, y_range = self._agent_side_ranges(name)
        if self._uses_capacity_layout():
            x_offset_range = self.capacity_goal_x_offset
            y_offset_range = self.capacity_goal_y_offset
        else:
            x_offset_range = (-1.8, 1.8)
            y_offset_range = (-2.2, 2.2)
        while True:
            x_offset = random.uniform(*x_offset_range)
            y_offset = random.uniform(*y_offset_range)
            candidate = np.array(
                [
                    np.clip(self.robot_positions[name][0] + x_offset, x_range[0], x_range[1]),
                    np.clip(self.robot_positions[name][1] + y_offset, y_range[0], y_range[1]),
                ]
            )
            if np.linalg.norm(candidate - self.robot_positions[name]) < self.goal_min_distance:
                continue
            return candidate

    def _sample_start_heading(self, name):
        if (
            self.scenario_mode in ("curriculum", "manifest")
            and not self._current_case_uses_standard_layout()
        ):
            agent_case = self.current_curriculum_case["agents"][name]
            if "heading" in agent_case:
                return float(agent_case["heading"])
        goal_offset = self.goal_positions[name] - self.robot_positions[name]
        goal_heading = math.atan2(goal_offset[1], goal_offset[0])
        return goal_heading + np.random.uniform(-0.2, 0.2)

    def _advance_fixed_physics(self, duration):
        if self.fixed_physics_step_size is None:
            raise RuntimeError("fixed physics stepping is not enabled")
        self._pause_fixed_physics()
        iterations = max(1, int(round(duration / self.fixed_physics_step_size)))
        start_time = self._wait_for_stable_sim_time()
        self.sim_clock_time = start_time
        target_time = start_time + iterations * self.fixed_physics_step_size
        overshoot_tolerance = max(0.01, 2.0 * self.fixed_physics_step_size)
        errors = []
        for _ in range(3):
            current_time = self._get_gazebo_sim_time()
            if current_time > target_time + overshoot_tolerance:
                raise RuntimeError(
                    "Gazebo fixed step overshot before retry: target=%.6f actual=%.6f"
                    % (target_time, current_time)
                )
            if current_time + self.fixed_physics_step_size >= target_time:
                return current_time
            remaining = max(
                1,
                int(
                    round(
                        (target_time - current_time)
                        / self.fixed_physics_step_size
                    )
                ),
            )
            result = subprocess.run(
                ["gz", "world", "--multi-step", str(remaining)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=15.0,
                check=False,
            )
            if result.returncode != 0:
                errors.append(result.stderr.strip())
            try:
                self._wait_for_sim_time_at_least(
                    target_time - self.fixed_physics_step_size
                )
            except TimeoutError as exc:
                errors.append(str(exc))
                continue
            self._pause_fixed_physics()
            current_time = self._wait_for_stable_sim_time()
            self.sim_clock_time = current_time
            if current_time > target_time + overshoot_tolerance:
                raise RuntimeError(
                    "Gazebo fixed step overshot target: target=%.6f actual=%.6f"
                    % (target_time, current_time)
                )
            if current_time + self.fixed_physics_step_size >= target_time:
                return current_time
        raise TimeoutError(
            "Gazebo did not complete %d fixed physics steps: "
            "current=%.6f target=%.6f errors=%s"
            % (
                iterations,
                self._get_gazebo_sim_time(),
                target_time,
                " | ".join(error for error in errors if error),
            )
        )

    def _wait_for_sim_time_at_least(self, minimum_time, timeout=5.0):
        deadline = time.monotonic() + timeout
        current_time = self._get_gazebo_sim_time()
        while current_time < minimum_time and time.monotonic() < deadline:
            time.sleep(0.001)
            current_time = self._get_gazebo_sim_time()
        if current_time < minimum_time:
            raise TimeoutError(
                "Gazebo simulation time did not reach %.6f; actual=%.6f"
                % (minimum_time, current_time)
            )
        return current_time

    def _get_gazebo_sim_time(self):
        rospy.wait_for_service("/gazebo/get_world_properties")
        try:
            world_properties = self.get_world_properties_service()
        except rospy.ServiceException as exc:
            raise RuntimeError("Could not read Gazebo simulation time") from exc
        if not world_properties.success:
            raise RuntimeError(
                "Could not read Gazebo simulation time: %s"
                % world_properties.status_message
            )
        return float(world_properties.sim_time)

    def _wait_for_stable_sim_time(self, stable_duration=0.02, timeout=5.0):
        deadline = time.monotonic() + timeout
        current_time = self._get_gazebo_sim_time()
        stable_since = time.monotonic()
        while time.monotonic() < deadline:
            time.sleep(0.005)
            next_time = self._get_gazebo_sim_time()
            if abs(next_time - current_time) > 0.5 * self.fixed_physics_step_size:
                current_time = next_time
                stable_since = time.monotonic()
                continue
            if time.monotonic() - stable_since >= stable_duration:
                return next_time
        raise TimeoutError("Gazebo simulation time did not become stable")

    def _reset_fixed_physics(self):
        self._pause_fixed_physics()
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()
        except rospy.ServiceException as exc:
            raise RuntimeError("Could not reset fixed-step Gazebo world") from exc
        self._pause_fixed_physics()
        self.sim_clock_time = self._wait_for_stable_sim_time()

    def _pause_fixed_physics(self):
        rospy.wait_for_service("/gazebo/pause_physics")
        rospy.wait_for_service("/gazebo/get_physics_properties")
        for _ in range(3):
            try:
                self.pause()
                physics = self.get_physics_properties_service()
            except rospy.ServiceException as exc:
                raise RuntimeError("Could not pause Gazebo for fixed stepping") from exc
            if physics.success and physics.pause:
                return
            time.sleep(0.01)
        raise RuntimeError("Gazebo did not enter the paused state")

    def _wait_for_fixed_agent_interfaces(self, timeout=30.0):
        if self.fixed_agent_interfaces_ready:
            return
        # Gazebo inserts spawned models on world updates. A server launched
        # paused can otherwise consume the first measured multi-step request
        # while robot plugins are still loading.
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException as exc:
            raise RuntimeError("Could not warm up fixed-step Gazebo") from exc
        deadline = time.monotonic() + timeout
        while True:
            world_properties = self.get_world_properties_service()
            missing_models = [
                name
                for name in self.agent_names
                if name not in world_properties.model_names
            ]
            missing_interfaces = [
                name
                for name in self.agent_names
                if self.vel_pubs[name].get_num_connections() < 1
                or self.velodyne_subscribers[self.agent_names.index(name)].get_num_connections()
                < 1
                or self.odom_subscribers[self.agent_names.index(name)].get_num_connections()
                < 1
            ]
            if not missing_models and not missing_interfaces:
                time.sleep(0.5)
                self._pause_fixed_physics()
                self._wait_for_stable_sim_time()
                try:
                    self.reset_simulation_proxy()
                except rospy.ServiceException as exc:
                    raise RuntimeError(
                        "Could not align fixed-step Gazebo simulation time"
                    ) from exc
                time.sleep(0.05)
                self._pause_fixed_physics()
                self._wait_for_stable_sim_time()
                self.fixed_agent_interfaces_ready = True
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for fixed-step agent interfaces: "
                    "models=%s interfaces=%s"
                    % (", ".join(missing_models), ", ".join(missing_interfaces))
                )
            time.sleep(0.05)

    def _wait_for_fixed_sensor_updates(
        self,
        previous_velodyne_counts,
        previous_odom_counts,
        target_time,
        timeout=5.0,
    ):
        # Velodyne publishes at 10 Hz, so the newest scan may precede the
        # requested 0.2 s boundary by almost 0.1 s. Reject older queued data,
        # especially messages left over across a simulation-time reset.
        oldest_allowed = float(target_time) - 0.11
        newest_allowed = float(target_time) + max(
            self.fixed_physics_step_size, 0.01
        )
        deadline = time.monotonic() + timeout
        while True:
            missing = [
                name
                for name in self.agent_names
                if self.velodyne_update_counts[name]
                <= previous_velodyne_counts[name]
                or self.odom_update_counts[name] <= previous_odom_counts[name]
                or not oldest_allowed
                <= self.velodyne_timestamps[name]
                <= newest_allowed
                or not oldest_allowed <= self.odom_timestamps[name] <= newest_allowed
            ]
            if not missing:
                return
            if time.monotonic() >= deadline:
                now = self.sim_clock_time
                sensor_details = []
                for name in self.agent_names:
                    sensor_details.append(
                        "%s[vel_count=%d->%d vel_stamp=%.6f; "
                        "odom_count=%d->%d odom_stamp=%.6f]"
                        % (
                            name,
                            previous_velodyne_counts[name],
                            self.velodyne_update_counts[name],
                            self.velodyne_timestamps[name],
                            previous_odom_counts[name],
                            self.odom_update_counts[name],
                            self.odom_timestamps[name],
                        )
                    )
                raise TimeoutError(
                    "Timed out waiting for fresh fixed-step sensors: %s | "
                    "target=%.6f now=%.6f allowed=[%.6f, %.6f] | %s"
                    % (
                        ", ".join(missing),
                        target_time,
                        now,
                        oldest_allowed,
                        newest_allowed,
                        " ".join(sensor_details),
                    )
                )
            time.sleep(0.001)

    def _publish_zero_commands(self, repeats=3):
        command = Twist()
        for _ in range(max(int(repeats), 1)):
            for publisher in self.vel_pubs.values():
                publisher.publish(command)
            time.sleep(0.02)

    def step(self, actions, active_mask=None):
        if active_mask is None:
            active_mask = [True] * self.num_agents

        commands = {}
        for idx, name in enumerate(self.agent_names):
            command = Twist()
            if active_mask[idx]:
                command.linear.x = actions[idx][0]
                command.angular.z = actions[idx][1]
            commands[name] = command
            self.vel_pubs[name].publish(command)

        self.publish_goal_markers()
        if self.fixed_physics_step_size is not None:
            for _ in range(3):
                time.sleep(0.02)
                for name, command in commands.items():
                    self.vel_pubs[name].publish(command)
            previous_velodyne_counts = dict(self.velodyne_update_counts)
            previous_odom_counts = dict(self.odom_update_counts)
            target_time = self._advance_fixed_physics(max(TIME_DELTA, 0.2))
            self._wait_for_fixed_sensor_updates(
                previous_velodyne_counts,
                previous_odom_counts,
                target_time,
            )
            self.wait_for_all_odom()
        else:
            rospy.wait_for_service("/gazebo/unpause_physics")
            try:
                self.unpause()
            except rospy.ServiceException:
                print("/gazebo/unpause_physics service call failed")

            time.sleep(max(TIME_DELTA, 0.2))
            self.wait_for_all_odom()

            rospy.wait_for_service("/gazebo/pause_physics")
            try:
                self.pause()
            except rospy.ServiceException:
                print("/gazebo/pause_physics service call failed")

        self._sync_robot_positions_from_odom()

        next_states = []
        rewards = []
        dones = []
        targets = []
        collisions = []
        step_agents_info = {}
        active_names = {
            name
            for idx, name in enumerate(self.agent_names)
            if idx < len(active_mask) and active_mask[idx]
        }

        for idx, name in enumerate(self.agent_names):
            state, distance = self._build_state(name, actions[idx])
            done, collision, min_laser = self.observe_collision(self.velodyne_data[name])
            # A collision at the goal boundary is a failure, not a simultaneous success.
            target = distance < GOAL_REACHED_DIST and not collision
            if target:
                done = True
            progress = (
                0.0
                if self.previous_distances[name] is None
                else self.previous_distances[name] - distance
            )
            nearest_robot_distance = self._nearest_visible_robot_distance(
                name, active_names
            )
            robot_threat = (
                self.robot_safe_distance > 0.0
                and np.isfinite(nearest_robot_distance)
                and nearest_robot_distance < self.robot_safe_distance
            )
            reward = self.get_reward(
                target,
                collision,
                actions[idx],
                min_laser,
                progress,
                suppress_stagnation=robot_threat,
                suppress_forward_reward=robot_threat,
            )
            anti_stagnation_penalty = self._compute_anti_stagnation_penalty(
                target,
                collision,
                actions[idx],
                min_laser,
                progress,
                nearest_robot_distance,
            )
            safe_recovery_reward = self._compute_safe_recovery_reward(
                target,
                collision,
                actions[idx],
                min_laser,
                progress,
                nearest_robot_distance,
            )
            reward += safe_recovery_reward - anti_stagnation_penalty
            wall_clearance_penalty = self._compute_wall_clearance_penalty(
                target, collision, actions[idx], min_laser
            )
            reward -= wall_clearance_penalty
            local_navigation_bonus = self._compute_local_navigation_bonus(
                target, collision, actions[idx], distance, state[-3], min_laser
            )
            reward += local_navigation_bonus
            self.previous_distances[name] = distance
            robot_proximity_penalty = self._compute_robot_proximity_penalty(
                actions[idx], nearest_robot_distance
            )
            reward -= robot_proximity_penalty
            robot_clearance_reward = self._compute_robot_clearance_reward(
                target,
                collision,
                self.previous_nearest_robot_distances[name],
                nearest_robot_distance,
            )
            reward += robot_clearance_reward
            yield_priority_reward = self._compute_yield_priority_reward(
                name,
                actions[idx],
                distance,
                active_names,
                target=target,
                collision=collision,
            )
            reward += yield_priority_reward

            if not active_mask[idx]:
                reward = 0.0
                done = True
                collision = False
                target = False
                progress = 0.0
                anti_stagnation_penalty = 0.0
                safe_recovery_reward = 0.0
                wall_clearance_penalty = 0.0
                local_navigation_bonus = 0.0
                robot_proximity_penalty = 0.0
                robot_clearance_reward = 0.0
                yield_priority_reward = 0.0
                nearest_robot_distance = None
            else:
                self.previous_nearest_robot_distances[name] = nearest_robot_distance

            next_states.append(state)
            rewards.append(reward)
            dones.append(done)
            targets.append(target)
            collisions.append(collision)
            step_agents_info[name] = {
                "target": target,
                "collision": collision,
                "distance": distance,
                "progress": progress,
                "min_laser": min_laser,
                "nearest_robot_distance": nearest_robot_distance,
                "reward": reward,
                "raw_reward": reward,
                "interaction_reward": 0.0,
                "anti_stagnation_reward": -(
                    anti_stagnation_penalty
                )
                + safe_recovery_reward,
                "wall_clearance_reward": -wall_clearance_penalty,
                "local_navigation_reward": local_navigation_bonus,
                "robot_proximity_reward": -robot_proximity_penalty,
                "robot_clearance_reward": robot_clearance_reward,
                "yield_priority_reward": yield_priority_reward,
                "reward_neighbors": [],
                "active_visible_neighbor_count": 0,
                "nearest_active_visible_neighbor_distance": None,
            }

        if self.cooperative_reward:
            rewards = self._apply_cooperative_reward(
                rewards, active_mask, step_agents_info
            )
            for idx, name in enumerate(self.agent_names):
                step_agents_info[name]["reward"] = rewards[idx]
                step_agents_info[name]["reward_neighbors"] = self.last_reward_neighbors[
                    name
                ]
                step_agents_info[name]["interaction_reward"] = (
                    self.last_interaction_rewards[name]
                )
                step_agents_info[name]["active_visible_neighbor_count"] = (
                    self.last_active_visible_neighbor_counts[name]
                )
                active_neighbor_distances = [
                    float(
                        np.linalg.norm(
                            self.robot_positions[other_name]
                            - self.robot_positions[name]
                        )
                    )
                    for other_name in self.last_reward_neighbors[name]
                ]
                if active_neighbor_distances:
                    step_agents_info[name][
                        "nearest_active_visible_neighbor_distance"
                    ] = min(active_neighbor_distances)

        active_neighbor_counts = [
            step_agents_info[name]["active_visible_neighbor_count"]
            for idx, name in enumerate(self.agent_names)
            if idx < len(active_mask) and active_mask[idx]
        ]
        interaction_rewards = [
            step_agents_info[name]["interaction_reward"]
            for idx, name in enumerate(self.agent_names)
            if idx < len(active_mask) and active_mask[idx]
        ]
        anti_stagnation_rewards = [
            step_agents_info[name]["anti_stagnation_reward"]
            for idx, name in enumerate(self.agent_names)
            if idx < len(active_mask) and active_mask[idx]
        ]
        wall_clearance_rewards = [
            step_agents_info[name]["wall_clearance_reward"]
            for idx, name in enumerate(self.agent_names)
            if idx < len(active_mask) and active_mask[idx]
        ]
        local_navigation_rewards = [
            step_agents_info[name]["local_navigation_reward"]
            for idx, name in enumerate(self.agent_names)
            if idx < len(active_mask) and active_mask[idx]
        ]
        robot_proximity_rewards = [
            step_agents_info[name]["robot_proximity_reward"]
            for idx, name in enumerate(self.agent_names)
            if idx < len(active_mask) and active_mask[idx]
        ]
        robot_clearance_rewards = [
            step_agents_info[name]["robot_clearance_reward"]
            for idx, name in enumerate(self.agent_names)
            if idx < len(active_mask) and active_mask[idx]
        ]
        yield_priority_rewards = [
            step_agents_info[name]["yield_priority_reward"]
            for idx, name in enumerate(self.agent_names)
            if idx < len(active_mask) and active_mask[idx]
        ]
        self.last_step_info = {
            "agents": step_agents_info,
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "success_count": int(sum(int(flag) for flag in targets)),
            "collision_count": int(sum(int(flag) for flag in collisions)),
            "active_neighbor_agent_count": int(
                sum(1 for count in active_neighbor_counts if count > 0)
            ),
            "mean_active_visible_neighbors": (
                float(np.mean(active_neighbor_counts)) if active_neighbor_counts else 0.0
            ),
            "max_active_visible_neighbors": (
                int(max(active_neighbor_counts)) if active_neighbor_counts else 0
            ),
            "mean_interaction_reward": (
                float(np.mean(interaction_rewards)) if interaction_rewards else 0.0
            ),
            "mean_anti_stagnation_reward": (
                float(np.mean(anti_stagnation_rewards))
                if anti_stagnation_rewards
                else 0.0
            ),
            "mean_wall_clearance_reward": (
                float(np.mean(wall_clearance_rewards))
                if wall_clearance_rewards
                else 0.0
            ),
            "mean_local_navigation_reward": (
                float(np.mean(local_navigation_rewards))
                if local_navigation_rewards
                else 0.0
            ),
            "mean_robot_proximity_reward": (
                float(np.mean(robot_proximity_rewards))
                if robot_proximity_rewards
                else 0.0
            ),
            "mean_robot_clearance_reward": (
                float(np.mean(robot_clearance_rewards))
                if robot_clearance_rewards
                else 0.0
            ),
            "mean_yield_priority_reward": (
                float(np.mean(yield_priority_rewards))
                if yield_priority_rewards
                else 0.0
            ),
        }

        return next_states, rewards, dones, targets, collisions

    def reset(self):
        if self.fixed_physics_step_size is not None:
            self._wait_for_fixed_agent_interfaces()
            self._reset_fixed_physics()
        self._publish_zero_commands()
        if self.fixed_physics_step_size is None:
            reset_service = "/gazebo/reset_world"
            rospy.wait_for_service(reset_service)
            try:
                self.reset_proxy()
            except rospy.ServiceException:
                print("%s service call failed" % reset_service)

        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004

        self.last_odom = {name: None for name in self.agent_names}
        self.velodyne_timestamps = {
            name: float("-inf") for name in self.agent_names
        }
        self.odom_timestamps = {name: float("-inf") for name in self.agent_names}
        self.previous_distances = {name: None for name in self.agent_names}
        self.previous_nearest_robot_distances = {
            name: None for name in self.agent_names
        }
        self.yield_wait_steps = {name: 0 for name in self.agent_names}
        self.yield_nearest_distances = {name: None for name in self.agent_names}
        self.last_step_info = self._empty_last_step_info()

        last_error = None
        for reset_attempt in range(20):
            try:
                if self.scenario_mode in ("curriculum", "manifest"):
                    self.current_curriculum_case = self._select_curriculum_case()
                spawn_positions = self._sample_robot_positions()
                for name, position in spawn_positions.items():
                    self.robot_positions[name] = np.array(position)
                self.goal_positions = self._sample_goal_positions()
                break
            except RuntimeError as exc:
                last_error = exc
                print(
                    "Reset placement retry %i/20 failed: %s"
                    % (reset_attempt + 1, exc)
                )
        else:
            raise RuntimeError(
                "Could not sample a valid multi-agent reset after 20 attempts. "
                f"Last error: {last_error}"
            )

        for name, position in spawn_positions.items():
            angle = self._sample_start_heading(name)
            quaternion = Quaternion.from_euler(0.0, 0.0, angle)
            state = self.set_self_states[name]
            state.pose.position.x = position[0]
            state.pose.position.y = position[1]
            state.pose.orientation.x = quaternion.x
            state.pose.orientation.y = quaternion.y
            state.pose.orientation.z = quaternion.z
            state.pose.orientation.w = quaternion.w
            self._set_model_state(state)

        if self.scenario_mode in ("curriculum", "manifest"):
            self._apply_curriculum_boxes()
        else:
            self.random_box()
        self.publish_goal_markers()
        if self.fixed_physics_step_size is not None:
            self._publish_zero_commands()
            time.sleep(0.05)
            previous_velodyne_counts = dict(self.velodyne_update_counts)
            previous_odom_counts = dict(self.odom_update_counts)
            target_time = self._advance_fixed_physics(max(TIME_DELTA, 0.2))
            self._wait_for_fixed_sensor_updates(
                previous_velodyne_counts,
                previous_odom_counts,
                target_time,
            )
        else:
            rospy.wait_for_service("/gazebo/unpause_physics")
            try:
                self.unpause()
            except rospy.ServiceException:
                print("/gazebo/unpause_physics service call failed")

            time.sleep(TIME_DELTA)

            rospy.wait_for_service("/gazebo/pause_physics")
            try:
                self.pause()
            except rospy.ServiceException:
                print("/gazebo/pause_physics service call failed")

        initial_states = []
        for name in self.agent_names:
            state, distance = self._build_state(name, [0.0, 0.0])
            self.previous_distances[name] = distance
            self.last_step_info["agents"][name]["distance"] = distance
            self.last_step_info["agents"][name]["min_laser"] = float(
                np.min(self.velodyne_data[name])
            )
            initial_states.append(state)
        for name in self.agent_names:
            nearest_robot_distance = self._nearest_visible_robot_distance(
                name, set(self.agent_names)
            )
            self.previous_nearest_robot_distances[name] = nearest_robot_distance
            self.last_step_info["agents"][name][
                "nearest_robot_distance"
            ] = nearest_robot_distance
        return initial_states

    def _sample_robot_positions(self, min_clearance=1.2):
        if (
            self.scenario_mode in ("curriculum", "manifest")
            and not self._current_case_uses_standard_layout()
        ):
            return {
                name: self._curriculum_agent_position(name, "start")
                for name in self.agent_names
            }
        if self.scenario_mode == "dense":
            min_clearance = self.dense_robot_clearance
        elif self._uses_capacity_layout():
            min_clearance = min(min_clearance, self.capacity_robot_clearance)
        if self.weak_coupling_layout:
            if self.num_agents <= 2:
                min_clearance = max(min_clearance, 3.0)
            elif not self._uses_capacity_layout():
                min_clearance = max(min_clearance, 1.2)
        positions = {}
        for name in self.agent_names:
            placed = False
            for _ in range(1000):
                x_range, y_range = self._agent_side_ranges(name)
                candidate = self._sample_position(x_range, y_range)
                if any(
                    np.linalg.norm(candidate - existing) < min_clearance
                    for existing in positions.values()
                ):
                    continue
                positions[name] = candidate
                placed = True
                break
            if not placed:
                raise RuntimeError(
                    f"Could not place robot {name} with clearance {min_clearance}. "
                    f"Placed {len(positions)}/{self.num_agents} robots. "
                    "The map may be too crowded for this robot count."
                )
        return positions

    def _sample_goal_positions(self, min_clearance=1.2):
        if (
            self.scenario_mode in ("curriculum", "manifest")
            and not self._current_case_uses_standard_layout()
        ):
            return {
                name: self._curriculum_agent_position(name, "goal")
                for name in self.agent_names
            }
        if self.scenario_mode == "dense":
            min_clearance = self.dense_goal_clearance
            goal_min_distance = self.dense_goal_min_distance
            goal_max_distance = self.dense_goal_max_distance
            robot_goal_clearance = self.dense_goal_clearance
        elif self._uses_capacity_layout():
            min_clearance = min(min_clearance, self.capacity_goal_clearance)
            goal_min_distance = self.goal_min_distance
            goal_max_distance = self.capacity_goal_max_distance
            robot_goal_clearance = self.capacity_robot_goal_clearance
        else:
            goal_min_distance = self.goal_min_distance
            goal_max_distance = self.goal_max_distance
            robot_goal_clearance = self.goal_clearance

        clearance_schedule = [min_clearance]
        if self.scenario_mode == "dense":
            clearance_schedule = [
                min_clearance,
                max(min_clearance * 0.75, 0.55),
                max(min_clearance * 0.6, 0.45),
            ]
        elif self.weak_coupling_layout:
            if self.num_agents <= 2:
                clearance_schedule = [max(min_clearance, 1.8), min_clearance]
            elif self._uses_capacity_layout():
                clearance_schedule = [
                    min_clearance,
                    max(min_clearance * 0.8, 0.65),
                    max(min_clearance * 0.65, 0.55),
                    max(min_clearance * 0.5, 0.45),
                ]
            else:
                clearance_schedule = [min_clearance, 1.0, 0.8]

        last_error = None
        for clearance in clearance_schedule:
            goals = {}
            for name in self.agent_names:
                placed = False
                for _ in range(2000):
                    candidate = self._sample_goal_candidate_for_agent(name)
                    if not check_pos(candidate[0], candidate[1]):
                        continue
                    goal_distance = np.linalg.norm(
                        candidate - self.robot_positions[name]
                    )
                    if (
                        goal_distance < goal_min_distance
                        or goal_distance > goal_max_distance
                    ):
                        continue
                    if any(
                        np.linalg.norm(candidate - other_robot) < robot_goal_clearance
                        for other_robot in self.robot_positions.values()
                    ):
                        continue
                    if any(
                        np.linalg.norm(candidate - existing_goal) < clearance
                        for existing_goal in goals.values()
                    ):
                        continue
                    goals[name] = candidate
                    placed = True
                    break
                if not placed:
                    last_error = (
                        f"Could not place goal for {name} with clearance {clearance}. "
                        f"Placed {len(goals)}/{self.num_agents} goals."
                    )
                    break
            if len(goals) == self.num_agents:
                if clearance < min_clearance:
                    print(
                        "Goal placement used relaxed clearance %.2f for %i agents"
                        % (clearance, self.num_agents)
                    )
                return goals

        raise RuntimeError(
            f"{last_error} The map may be too crowded for this robot count."
        )

    def random_box(self):
        for i in range(4):
            name = "cardboard_box_" + str(i)
            box_ok = False
            while not box_ok:
                candidate = np.array(
                    [np.random.uniform(-6, 6), np.random.uniform(-6, 6)]
                )
                if not check_pos(candidate[0], candidate[1]):
                    continue

                if self._uses_capacity_layout():
                    clearance = 1.0
                else:
                    clearance = 2.0 if self.weak_coupling_layout else 1.5
                too_close_robot = any(
                    np.linalg.norm(candidate - robot_pos) < clearance
                    for robot_pos in self.robot_positions.values()
                )
                too_close_goal = any(
                    np.linalg.norm(candidate - goal_pos) < clearance
                    for goal_pos in self.goal_positions.values()
                )
                if too_close_robot or too_close_goal:
                    continue
                box_ok = True

            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = candidate[0]
            box_state.pose.position.y = candidate[1]
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self._set_model_state(box_state)

    def publish_goal_markers(self):
        marker_array = MarkerArray()
        for idx, name in enumerate(self.agent_names):
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.ns = "goal_points"
            marker.id = idx
            marker.type = marker.CYLINDER
            marker.action = marker.ADD
            marker.scale.x = 0.15
            marker.scale.y = 0.15
            marker.scale.z = 0.02
            marker.color.a = 1.0
            color = AGENT_COLORS[idx % len(AGENT_COLORS)]
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = self.goal_positions[name][0]
            marker.pose.position.y = self.goal_positions[name][1]
            marker.pose.position.z = 0.0
            marker_array.markers.append(marker)
        self.goal_publisher.publish(marker_array)

    @staticmethod
    def observe_collision(laser_data):
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser

    def get_reward(
        self,
        target,
        collision,
        action,
        min_laser,
        progress,
        suppress_stagnation=False,
        suppress_forward_reward=False,
    ):
        if target:
            return 100.0
        if collision:
            return -100.0
        obstacle_penalty = 1 - min_laser if min_laser < 1 else 0.0
        progress_reward = 20.0 * progress
        forward_reward = (
            0.0
            if suppress_forward_reward
            else self.forward_reward_weight * action[0]
        )
        turn_penalty = 0.2 * abs(action[1])
        stagnation_penalty = (
            self.stagnation_penalty_weight
            if (
                not suppress_stagnation
                and action[0] < 0.1
                and abs(progress) < 0.01
            )
            else 0.0
        )
        return (
            progress_reward
            + forward_reward
            - turn_penalty
            - 0.5 * obstacle_penalty
            - stagnation_penalty
        )
