import json
import os
import random
import socket
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from actor_models import (
    Actor,
    ResidualActor,
    actor_hidden_dims_from_state_dict,
    is_residual_actor_state_dict,
)
from interaction_oracle import interaction_mask
from learned_gate_controller import LearnedInteractionGateController
from multi_agent_velodyne_env import MultiAgentGazeboEnv
from oracle_controllers import ConflictPairYieldOracle, RightHandPassOracle
from outcome_utils import resolve_terminal_outcome
from robot_perception.recorder import PerceptionShardRecorder
from rule_gate_controllers import MinLidarActorSwitcher


class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        actor_mode="full",
        residual_hidden_dim=128,
        residual_scale=0.15,
    ):
        if actor_mode == "residual":
            self.actor = ResidualActor(
                state_dim,
                action_dim,
                hidden_dim=residual_hidden_dim,
                residual_scale=residual_scale,
            ).to(device)
        elif actor_mode in ("full", "head_only"):
            self.actor = Actor(state_dim, action_dim).to(device)
        else:
            raise ValueError("Unsupported test actor mode: %s" % actor_mode)
        self.actor_mode = actor_mode
        self.residual_scale = float(residual_scale)

    def get_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def load(self, filename, directory):
        actor_state = torch.load(
            "%s/%s_actor.pth" % (directory, filename), map_location=device
        )
        residual_checkpoint = is_residual_actor_state_dict(actor_state)
        if self.actor_mode == "residual" and not residual_checkpoint:
            raise ValueError("Residual test mode requires a residual actor checkpoint")
        if self.actor_mode != "residual" and residual_checkpoint:
            raise ValueError("Residual actor checkpoint requires residual test mode")
        if self.actor_mode != "residual":
            hidden_dim_1, hidden_dim_2 = actor_hidden_dims_from_state_dict(actor_state)
            current_hidden_dims = (
                self.actor.hidden_dim_1,
                self.actor.hidden_dim_2,
            )
            if (hidden_dim_1, hidden_dim_2) != current_hidden_dims:
                self.actor = Actor(
                    self.actor.layer_1.in_features,
                    self.actor.layer_3.out_features,
                    hidden_dim_1=hidden_dim_1,
                    hidden_dim_2=hidden_dim_2,
                ).to(device)
        self.actor.load_state_dict(actor_state)
        if self.actor_mode == "residual":
            self.residual_scale = self.actor.residual_scale


class DualActorSwitcher(object):
    def __init__(
        self,
        standard_policy,
        dense_policy,
        switch_on_distance,
        switch_off_distance,
        switch_on_visible_neighbors,
    ):
        self.standard_policy = standard_policy
        self.dense_policy = dense_policy
        self.switch_on_distance = float(switch_on_distance)
        self.switch_off_distance = max(
            float(switch_off_distance), float(switch_on_distance)
        )
        self.switch_on_visible_neighbors = max(int(switch_on_visible_neighbors), 1)
        self.current_mode = {}

    def reset(self, agent_names):
        self.current_mode = {name: "standard" for name in agent_names}

    def _nearest_visible_neighbor_distance(self, env, name):
        visible_neighbors = env._compute_visible_neighbors(name)
        if not visible_neighbors:
            return None, 0
        origin = env.robot_positions[name]
        distances = [
            float(np.linalg.norm(env.robot_positions[other_name] - origin))
            for other_name in visible_neighbors
        ]
        return min(distances), len(visible_neighbors)

    def choose_action(self, env, name, state):
        nearest_distance, visible_count = self._nearest_visible_neighbor_distance(
            env, name
        )
        mode = self.current_mode.get(name, "standard")

        should_switch_dense = (
            visible_count >= self.switch_on_visible_neighbors
            and nearest_distance is not None
            and nearest_distance <= self.switch_on_distance
        )
        should_switch_standard = (
            nearest_distance is None or nearest_distance >= self.switch_off_distance
        )

        if mode == "standard" and should_switch_dense:
            mode = "dense"
        elif mode == "dense" and should_switch_standard:
            mode = "standard"

        self.current_mode[name] = mode
        policy = self.dense_policy if mode == "dense" else self.standard_policy
        action = policy.get_action(np.array(state))
        return action, mode, nearest_distance, visible_count


class CaseOracleSwitcher(object):
    def __init__(self, standard_policy, dense_policy, case_actor_map):
        self.standard_policy = standard_policy
        self.dense_policy = dense_policy
        self.case_actor_map = dict(case_actor_map)

    def reset(self, agent_names):
        return None

    def choose_action(self, env, name, state):
        case = getattr(env, "current_curriculum_case", None)
        case_name = "standard"
        if isinstance(case, dict):
            case_name = str(case.get("name") or "unnamed_curriculum_case")
        mode = self.case_actor_map.get(case_name, self.case_actor_map.get("default", "standard"))
        if mode not in ("standard", "dense"):
            raise ValueError(
                "Case oracle mode must map cases to 'standard' or 'dense', got %r for %s"
                % (mode, case_name)
            )
        policy = self.dense_policy if mode == "dense" else self.standard_policy
        action = policy.get_action(np.array(state))
        return action, mode, None, None


class RecoveryOracleSwitcher(object):
    def __init__(
        self,
        standard_policy,
        dense_policy,
        candidate_distance,
        release_distance,
        progress_threshold,
        progress_window,
        distance_delta_threshold,
        goal_distance,
        minimum_hold_steps,
        maximum_hold_steps,
    ):
        self.standard_policy = standard_policy
        self.dense_policy = dense_policy
        self.candidate_distance = float(candidate_distance)
        self.release_distance = max(float(release_distance), self.candidate_distance)
        self.progress_threshold = float(progress_threshold)
        self.progress_window = max(int(progress_window), 1)
        self.distance_delta_threshold = float(distance_delta_threshold)
        self.goal_distance = float(goal_distance)
        self.minimum_hold_steps = max(int(minimum_hold_steps), 0)
        self.maximum_hold_steps = max(int(maximum_hold_steps), 0)
        self.current_mode = {}
        self.hold_steps = {}
        self.progress_history = {}
        self.distance_history = {}

    def reset(self, agent_names):
        self.current_mode = {name: "standard" for name in agent_names}
        self.hold_steps = {name: 0 for name in agent_names}
        self.progress_history = {name: [] for name in agent_names}
        self.distance_history = {name: [] for name in agent_names}

    def _nearest_active_visible_neighbor_distance(self, env, name):
        active_names = {
            other_name
            for other_name, info in env.last_step_info["agents"].items()
            if other_name != name
            and not bool(info["target"])
            and not bool(info["collision"])
        }
        visible_neighbors = [
            other_name
            for other_name in env._compute_visible_neighbors(name)
            if other_name in active_names
        ]
        if not visible_neighbors:
            return None, 0
        origin = env.robot_positions[name]
        distances = [
            float(np.linalg.norm(env.robot_positions[other_name] - origin))
            for other_name in visible_neighbors
        ]
        return min(distances), len(visible_neighbors)

    def _update_history(self, name, info):
        progress = float(info.get("progress") or 0.0)
        distance = info.get("distance")
        if distance is not None:
            distance = float(distance)
        progress_values = self.progress_history.setdefault(name, [])
        distance_values = self.distance_history.setdefault(name, [])
        progress_values.append(progress)
        if distance is not None:
            distance_values.append(distance)
        if len(progress_values) > self.progress_window:
            del progress_values[0 : len(progress_values) - self.progress_window]
        if len(distance_values) > self.progress_window + 1:
            del distance_values[0 : len(distance_values) - self.progress_window - 1]
        return progress_values, distance_values, distance

    def choose_action(self, env, name, state):
        info = env.last_step_info["agents"].get(name, {})
        progress_values, distance_values, distance = self._update_history(name, info)
        nearest_distance, visible_count = self._nearest_active_visible_neighbor_distance(
            env, name
        )
        mode = self.current_mode.get(name, "standard")
        hold_steps = self.hold_steps.get(name, 0)

        near_candidate = (
            nearest_distance is not None
            and nearest_distance <= self.candidate_distance
            and visible_count > 0
        )
        released = nearest_distance is None or nearest_distance >= self.release_distance
        near_goal = distance is not None and distance <= self.goal_distance
        mean_progress = (
            float(np.mean(progress_values)) if progress_values else 0.0
        )
        has_progress_window = len(progress_values) >= self.progress_window
        low_progress = (
            has_progress_window and mean_progress <= self.progress_threshold
        )
        distance_stagnant = False
        if len(distance_values) >= self.progress_window + 1:
            distance_delta = distance_values[0] - distance_values[-1]
            distance_stagnant = distance_delta <= self.distance_delta_threshold
        stagnating = low_progress or distance_stagnant

        should_switch_recovery = near_candidate and stagnating and not near_goal
        should_switch_standard = released or near_goal or not stagnating

        if mode == "standard" and should_switch_recovery:
            mode = "dense"
            hold_steps = 0
        elif mode == "dense":
            hold_steps += 1
            if self.maximum_hold_steps and hold_steps >= self.maximum_hold_steps:
                mode = "standard"
                hold_steps = 0
            elif hold_steps >= self.minimum_hold_steps and should_switch_standard:
                mode = "standard"
                hold_steps = 0

        self.current_mode[name] = mode
        self.hold_steps[name] = hold_steps
        policy = self.dense_policy if mode == "dense" else self.standard_policy
        action = policy.get_action(np.array(state))
        diagnostics = {
            "nearest_distance": nearest_distance,
            "visible_count": visible_count,
            "mean_progress": mean_progress,
            "low_progress": low_progress,
            "distance_stagnant": distance_stagnant,
            "near_candidate": near_candidate,
            "near_goal": near_goal,
            "hold_steps": hold_steps,
        }
        return action, mode, None, diagnostics


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def env_int(name, default):
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def env_float(name, default):
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def env_json_path(name):
    value = os.environ.get(name)
    if value is None:
        return ""
    return value.strip()


def make_agent_names():
    explicit_names = os.environ.get("DRL_MULTI_AGENT_NAMES")
    if explicit_names and explicit_names.strip():
        names = [name.strip() for name in explicit_names.split(",") if name.strip()]
        if not names:
            raise ValueError("DRL_MULTI_AGENT_NAMES did not contain valid agent names")
        return names

    num_agents = env_int("DRL_MULTI_NUM_AGENTS", 2)
    if num_agents < 1 or num_agents > 10:
        raise ValueError("DRL_MULTI_NUM_AGENTS must be between 1 and 10")
    return [f"r{idx}" for idx in range(1, num_agents + 1)]


seed = env_int("DRL_MULTI_SEED", 0)
max_ep = 300
target_test_episodes = int(os.environ.get("DRL_MULTI_TEST_TARGET_EPISODES", "0"))
scenario_mode = os.environ.get("DRL_MULTI_SCENARIO", "standard").strip().lower()
base_file_name = "TD3_velodyne_multi_v4"
file_name = os.environ.get("DRL_MULTI_TEST_FILE_NAME", base_file_name)
actor_mode = os.environ.get("DRL_MULTI_TEST_ACTOR_MODE", "full").strip().lower()
residual_hidden_dim = env_int("DRL_MULTI_RESIDUAL_HIDDEN_DIM", 128)
residual_scale = env_float("DRL_MULTI_RESIDUAL_SCALE", 0.15)
standard_actor_file = os.environ.get(
    "DRL_MULTI_STANDARD_ACTOR_FILE", file_name
).strip()
dense_actor_file = os.environ.get("DRL_MULTI_DENSE_ACTOR_FILE", "").strip()
dense_actor_mode = os.environ.get(
    "DRL_MULTI_DENSE_ACTOR_MODE", "full"
).strip().lower()
actor_selection_mode = os.environ.get("DRL_MULTI_ACTOR_SELECTION_MODE", "").strip().lower()
dual_actor_enabled = bool(dense_actor_file)
if not actor_selection_mode:
    actor_selection_mode = "hard_switch" if dual_actor_enabled else "single"
if actor_selection_mode not in (
    "single",
    "hard_switch",
    "case_oracle",
    "interaction_oracle",
    "recovery_oracle",
    "learned_gate",
    "min_lidar_gate",
):
    raise ValueError(
        "DRL_MULTI_ACTOR_SELECTION_MODE must be one of: single, hard_switch, "
        "case_oracle, interaction_oracle, recovery_oracle, learned_gate, "
        "min_lidar_gate"
    )
if actor_selection_mode != "single" and not dual_actor_enabled:
    raise ValueError(
        "DRL_MULTI_DENSE_ACTOR_FILE is required when actor selection mode is not 'single'"
    )
switch_on_distance = env_float("DRL_MULTI_SWITCH_ON_DISTANCE", 1.6)
switch_off_distance = env_float("DRL_MULTI_SWITCH_OFF_DISTANCE", 2.0)
switch_on_visible_neighbors = env_int("DRL_MULTI_SWITCH_ON_VISIBLE_NEIGHBORS", 1)
gate_checkpoint_path = env_json_path("DRL_MULTI_GATE_CHECKPOINT")
gate_detector_checkpoint_path = env_json_path(
    "DRL_MULTI_GATE_DETECTOR_CHECKPOINT"
)
gate_switch_on_threshold = os.environ.get("DRL_MULTI_GATE_SWITCH_ON_THRESHOLD")
gate_switch_off_threshold = os.environ.get("DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD")
gate_switch_on_threshold = (
    float(gate_switch_on_threshold) if gate_switch_on_threshold else None
)
gate_switch_off_threshold = (
    float(gate_switch_off_threshold) if gate_switch_off_threshold else None
)
gate_minimum_hold_steps = env_int("DRL_MULTI_GATE_MINIMUM_HOLD_STEPS", 3)
gate_max_candidates = env_int("DRL_MULTI_GATE_MAX_CANDIDATES", 12)
gate_evaluation_stride = env_int("DRL_MULTI_GATE_EVALUATION_STRIDE", 1)
min_lidar_switch_on_distance = env_float(
    "DRL_MULTI_MIN_LIDAR_SWITCH_ON_DISTANCE", 2.0
)
min_lidar_switch_off_distance = env_float(
    "DRL_MULTI_MIN_LIDAR_SWITCH_OFF_DISTANCE", 2.2
)
min_lidar_minimum_hold_steps = env_int(
    "DRL_MULTI_MIN_LIDAR_MINIMUM_HOLD_STEPS", 3
)
if actor_selection_mode == "learned_gate":
    if not gate_checkpoint_path or not gate_detector_checkpoint_path:
        raise ValueError(
            "learned_gate mode requires DRL_MULTI_GATE_CHECKPOINT and "
            "DRL_MULTI_GATE_DETECTOR_CHECKPOINT"
        )
    os.environ["DRL_MULTI_RECORD_RAW_LIDAR"] = "1"
interaction_oracle_distance = env_float(
    "DRL_MULTI_ORACLE_INTERACTION_DISTANCE", 2.0
)
if interaction_oracle_distance <= 0.0:
    raise ValueError("DRL_MULTI_ORACLE_INTERACTION_DISTANCE must be positive")
recovery_oracle_candidate_distance = env_float(
    "DRL_MULTI_RECOVERY_ORACLE_CANDIDATE_DISTANCE", 2.0
)
recovery_oracle_release_distance = env_float(
    "DRL_MULTI_RECOVERY_ORACLE_RELEASE_DISTANCE", 2.4
)
recovery_oracle_progress_threshold = env_float(
    "DRL_MULTI_RECOVERY_ORACLE_PROGRESS_THRESHOLD", 0.003
)
recovery_oracle_progress_window = env_int(
    "DRL_MULTI_RECOVERY_ORACLE_PROGRESS_WINDOW", 5
)
recovery_oracle_distance_delta_threshold = env_float(
    "DRL_MULTI_RECOVERY_ORACLE_DISTANCE_DELTA_THRESHOLD", 0.02
)
recovery_oracle_goal_distance = env_float(
    "DRL_MULTI_RECOVERY_ORACLE_GOAL_DISTANCE", 0.45
)
recovery_oracle_minimum_hold_steps = env_int(
    "DRL_MULTI_RECOVERY_ORACLE_MINIMUM_HOLD_STEPS", 3
)
recovery_oracle_maximum_hold_steps = env_int(
    "DRL_MULTI_RECOVERY_ORACLE_MAXIMUM_HOLD_STEPS", 20
)
case_oracle_map_path = env_json_path("DRL_MULTI_CASE_ORACLE_MAP")
rule_oracle_mode = os.environ.get("DRL_MULTI_RULE_ORACLE_MODE", "").strip().lower()
if rule_oracle_mode not in ("", "conflict_pair_yield", "right_hand_pass"):
    raise ValueError(
        "DRL_MULTI_RULE_ORACLE_MODE must be empty, 'conflict_pair_yield', "
        "or 'right_hand_pass'"
    )
rule_oracle_schedule = os.environ.get(
    "DRL_MULTI_RULE_ORACLE_SCHEDULE", "all"
).strip().lower()
if rule_oracle_schedule not in ("all", "paired_alternating"):
    raise ValueError(
        "DRL_MULTI_RULE_ORACLE_SCHEDULE must be 'all' or 'paired_alternating'"
    )
rule_oracle_stop_distance = env_float("DRL_MULTI_RULE_ORACLE_STOP_DISTANCE", 1.2)
rule_oracle_release_distance = env_float("DRL_MULTI_RULE_ORACLE_RELEASE_DISTANCE", 1.4)
rule_oracle_max_yield_steps = env_int("DRL_MULTI_RULE_ORACLE_MAX_YIELD_STEPS", 20)
right_hand_activation_distance = env_float(
    "DRL_MULTI_RIGHT_HAND_ACTIVATION_DISTANCE", 1.5
)
right_hand_release_distance = env_float(
    "DRL_MULTI_RIGHT_HAND_RELEASE_DISTANCE", 1.8
)
right_hand_frontal_angle = env_float("DRL_MULTI_RIGHT_HAND_FRONTAL_ANGLE_DEG", 35.0)
right_hand_opposing_angle = env_float(
    "DRL_MULTI_RIGHT_HAND_OPPOSING_ANGLE_DEG", 150.0
)
right_hand_min_closing_speed = env_float(
    "DRL_MULTI_RIGHT_HAND_MIN_CLOSING_SPEED", 0.2
)
right_hand_max_ttc = env_float("DRL_MULTI_RIGHT_HAND_MAX_TTC", 3.0)
right_hand_turn_action = env_float("DRL_MULTI_RIGHT_HAND_TURN_ACTION", -0.6)
right_hand_linear_speed_cap = env_float(
    "DRL_MULTI_RIGHT_HAND_LINEAR_SPEED_CAP", 0.45
)
right_hand_max_override_steps = env_int(
    "DRL_MULTI_RIGHT_HAND_MAX_OVERRIDE_STEPS", 20
)
fixed_physics_step_size = env_float("DRL_MULTI_FIXED_PHYSICS_STEP_SIZE", None)
launchfile = os.environ.get(
    "DRL_MULTI_TEST_LAUNCHFILE", "multi_robot_scenario_multi_2.launch"
)
resume_testing = True
default_state_path = (
    "./checkpoints/TD3_velodyne_multi_test_state.pt"
    if file_name == base_file_name
    else f"./checkpoints/{file_name}_test_state.pt"
)
default_test_stats_path = (
    "./results/TD3_velodyne_multi_test.npy"
    if file_name == base_file_name
    else f"./results/{file_name}_test.npy"
)
state_path = os.environ.get(
    "DRL_MULTI_TEST_STATE_PATH",
    default_state_path,
)
test_stats_path = os.environ.get(
    "DRL_MULTI_TEST_STATS_PATH",
    default_test_stats_path,
)
trajectory_path = os.environ.get("DRL_MULTI_TRAJECTORY_JSONL", "").strip()
perception_output_dir = os.environ.get(
    "DRL_MULTI_ROBOT_PERCEPTION_OUTPUT_DIR", ""
).strip()
perception_split = os.environ.get(
    "DRL_MULTI_ROBOT_PERCEPTION_SPLIT", "train"
).strip()
perception_frame_stride = env_int("DRL_MULTI_ROBOT_PERCEPTION_FRAME_STRIDE", 2)
perception_max_background = env_int(
    "DRL_MULTI_ROBOT_PERCEPTION_MAX_BACKGROUND", 12
)
perception_run_metadata_path = env_json_path(
    "DRL_MULTI_ROBOT_PERCEPTION_RUN_METADATA_PATH"
)
perception_run_metadata = {}
if perception_run_metadata_path:
    with open(perception_run_metadata_path, "r", encoding="utf-8") as handle:
        perception_run_metadata = json.load(handle)
    if not isinstance(perception_run_metadata, dict):
        raise ValueError("robot-perception run metadata must be a JSON object")
if perception_output_dir:
    if scenario_mode != "manifest":
        raise ValueError("robot-perception recording requires scenario_mode=manifest")
    os.environ["DRL_MULTI_RECORD_RAW_LIDAR"] = "1"
print_every_episodes = 10
environment_dim = 20
robot_dim = 4
agent_names = make_agent_names()


def make_test_run_dir():
    timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
    return os.path.join("runs", f"test_{file_name}_{timestamp}_{socket.gethostname()}")


def load_test_state():
    if not (resume_testing and os.path.exists(state_path)):
        return None
    return torch.load(state_path, map_location="cpu")


def save_test_state(payload):
    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(payload, state_path)


def append_stats(record):
    os.makedirs("./results", exist_ok=True)
    if os.path.exists(test_stats_path):
        history = list(np.load(test_stats_path, allow_pickle=True))
    else:
        history = []
    history.append(record)
    np.save(test_stats_path, np.array(history, dtype=object))


def append_trajectory_step(record):
    if not trajectory_path:
        return
    directory = os.path.dirname(os.path.abspath(trajectory_path))
    os.makedirs(directory, exist_ok=True)
    with open(trajectory_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def current_case_name(env):
    case = getattr(env, "current_curriculum_case", None)
    if isinstance(case, dict):
        return str(case.get("scenario_id") or case.get("name") or "unnamed_curriculum_case")
    return "standard"


def current_case_perception_metadata(env):
    case = getattr(env, "current_curriculum_case", None)
    if not isinstance(case, dict):
        return "standard", "unknown"
    view = case.get("view", {})
    return (
        str(view.get("perception_pool") or case.get("preset") or "unknown"),
        str(view.get("interaction_band") or "unknown"),
    )


def new_case_record():
    return {
        "episodes": 0,
        "success": 0,
        "collision": 0,
        "unresolved": 0,
        "full_success": 0,
        "timeout": 0,
        "env_steps": 0,
        "final_distance_sum": 0.0,
    }


def update_case_stats(
    case_stats,
    case_name,
    episode_success_count,
    episode_collision_count,
    episode_unresolved_count,
    full_success,
    timeout_episode,
    episode_env_steps,
    mean_final_distance,
):
    stats = case_stats.setdefault(case_name, new_case_record())
    stats["episodes"] += 1
    stats["success"] += int(episode_success_count)
    stats["collision"] += int(episode_collision_count)
    stats["unresolved"] += int(episode_unresolved_count)
    stats["full_success"] += int(full_success)
    stats["timeout"] += int(timeout_episode)
    stats["env_steps"] += int(episode_env_steps)
    stats["final_distance_sum"] += float(mean_final_distance)


def print_case_stats(case_stats):
    if not case_stats:
        return
    print("Case summary:")
    for name in sorted(case_stats):
        stats = case_stats[name]
        episodes = max(int(stats["episodes"]), 1)
        denom = episodes * len(agent_names)
        avg_steps = stats["env_steps"] / episodes
        avg_final_distance = stats["final_distance_sum"] / episodes
        print(
            "  %s | episodes=%i | success_rate=%.3f | collision_rate=%.3f | "
            "unresolved_rate=%.3f | full_success_rate=%.3f | timeout_rate=%.3f | "
            "avg_env_steps=%.1f | avg_final_distance=%.3f"
            % (
                name,
                stats["episodes"],
                stats["success"] / denom,
                stats["collision"] / denom,
                stats["unresolved"] / denom,
                stats["full_success"] / episodes,
                stats["timeout"] / episodes,
                avg_steps,
                avg_final_distance,
            )
        )


def load_case_oracle_map():
    if actor_selection_mode != "case_oracle":
        return {}
    if not case_oracle_map_path:
        raise ValueError(
            "DRL_MULTI_CASE_ORACLE_MAP must point to a JSON file in case_oracle mode"
        )
    with open(case_oracle_map_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Case oracle map JSON must be a non-empty object")
    normalized = {}
    for key, value in payload.items():
        mode = str(value).strip().lower()
        if mode not in ("standard", "dense"):
            raise ValueError(
                "Case oracle map values must be 'standard' or 'dense', got %r for %s"
                % (value, key)
            )
        normalized[str(key)] = mode
    return normalized


env = MultiAgentGazeboEnv(
    launchfile,
    environment_dim,
    agent_names=agent_names,
    cooperative_reward=False,
    robot_safe_distance=0.0,
    weak_coupling_layout=True,
    scenario_mode=scenario_mode,
    active_neighbors_only=actor_selection_mode
    in ("interaction_oracle", "recovery_oracle"),
    fixed_physics_step_size=fixed_physics_step_size,
)
time.sleep(5)
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + robot_dim
action_dim = 2

network = TD3(
    state_dim,
    action_dim,
    actor_mode=actor_mode,
    residual_hidden_dim=residual_hidden_dim,
    residual_scale=residual_scale,
)
try:
    network.load(standard_actor_file, "./pytorch_models")
except Exception:
    raise ValueError("Could not load the stored multi-agent model parameters")

dense_network = None
dense_policy_controller = None
rule_oracle_controller = None
case_oracle_map = {}
if dual_actor_enabled:
    dense_network = TD3(
        state_dim,
        action_dim,
        actor_mode=dense_actor_mode,
        residual_hidden_dim=residual_hidden_dim,
        residual_scale=residual_scale,
    )
    try:
        dense_network.load(dense_actor_file, "./pytorch_models")
    except Exception:
        raise ValueError("Could not load the stored dense-actor parameters")
    if actor_selection_mode == "hard_switch":
        dense_policy_controller = DualActorSwitcher(
            standard_policy=network,
            dense_policy=dense_network,
            switch_on_distance=switch_on_distance,
            switch_off_distance=switch_off_distance,
            switch_on_visible_neighbors=switch_on_visible_neighbors,
        )
    elif actor_selection_mode == "case_oracle":
        case_oracle_map = load_case_oracle_map()
        dense_policy_controller = CaseOracleSwitcher(
            standard_policy=network,
            dense_policy=dense_network,
            case_actor_map=case_oracle_map,
        )
    elif actor_selection_mode == "learned_gate":
        dense_policy_controller = LearnedInteractionGateController(
            standard_policy=network,
            dense_policy=dense_network,
            detector_checkpoint=gate_detector_checkpoint_path,
            gate_checkpoint=gate_checkpoint_path,
            device=device,
            switch_on_threshold=gate_switch_on_threshold,
            switch_off_threshold=gate_switch_off_threshold,
            minimum_hold_steps=gate_minimum_hold_steps,
            max_candidates=gate_max_candidates,
            evaluation_stride=gate_evaluation_stride,
        )
    elif actor_selection_mode == "recovery_oracle":
        dense_policy_controller = RecoveryOracleSwitcher(
            standard_policy=network,
            dense_policy=dense_network,
            candidate_distance=recovery_oracle_candidate_distance,
            release_distance=recovery_oracle_release_distance,
            progress_threshold=recovery_oracle_progress_threshold,
            progress_window=recovery_oracle_progress_window,
            distance_delta_threshold=recovery_oracle_distance_delta_threshold,
            goal_distance=recovery_oracle_goal_distance,
            minimum_hold_steps=recovery_oracle_minimum_hold_steps,
            maximum_hold_steps=recovery_oracle_maximum_hold_steps,
        )
    elif actor_selection_mode == "min_lidar_gate":
        dense_policy_controller = MinLidarActorSwitcher(
            standard_policy=network,
            interaction_policy=dense_network,
            lidar_bins=environment_dim,
            switch_on_distance=min_lidar_switch_on_distance,
            switch_off_distance=min_lidar_switch_off_distance,
            minimum_hold_steps=min_lidar_minimum_hold_steps,
        )
if rule_oracle_mode == "conflict_pair_yield":
    if actor_selection_mode != "single":
        raise ValueError("Conflict-pair yield oracle only supports single actor mode")
    rule_oracle_controller = ConflictPairYieldOracle(
        base_policy=network,
        stop_distance=rule_oracle_stop_distance,
        release_distance=rule_oracle_release_distance,
        max_yield_steps=rule_oracle_max_yield_steps,
    )
elif rule_oracle_mode == "right_hand_pass":
    if actor_selection_mode != "single":
        raise ValueError("Right-hand pass oracle only supports single actor mode")
    rule_oracle_controller = RightHandPassOracle(
        base_policy=network,
        activation_distance=right_hand_activation_distance,
        release_distance=right_hand_release_distance,
        frontal_angle_degrees=right_hand_frontal_angle,
        opposing_angle_degrees=right_hand_opposing_angle,
        min_closing_speed=right_hand_min_closing_speed,
        max_ttc=right_hand_max_ttc,
        turn_action=right_hand_turn_action,
        linear_speed_cap=right_hand_linear_speed_cap,
        max_override_steps=right_hand_max_override_steps,
    )

test_state = load_test_state() or {}
if scenario_mode == "manifest":
    env.restore_manifest_sampling_state(test_state.get("manifest_sampling_state"))
episode_num = test_state.get("episode_num", 0)
total_env_steps = test_state.get("total_env_steps", 0)
total_agent_samples = test_state.get("total_agent_samples", 0)
success_count = test_state.get("success_count", 0)
collision_count = test_state.get("collision_count", 0)
unresolved_count = test_state.get("unresolved_count", 0)
full_success_count = test_state.get("full_success_count", 0)
timeout_episode_count = test_state.get("timeout_episode_count", 0)
success_hist = test_state.get("success_hist", [0] * (len(agent_names) + 1))
collision_hist = test_state.get("collision_hist", [0] * (len(agent_names) + 1))
case_stats = test_state.get("case_stats", {})
recent_rewards = []
recent_success_rates = []
recent_collision_rates = []
recent_unresolved_rates = []
recent_full_success = []
recent_timeout_episodes = []
log_dir = make_test_run_dir()
writer = SummaryWriter(log_dir=log_dir)
perception_recorder = (
    PerceptionShardRecorder(
        perception_output_dir,
        perception_split,
        frame_stride=perception_frame_stride,
        max_background_candidates=perception_max_background,
        actor_state_dim=state_dim,
        oracle_interaction_distance=interaction_oracle_distance,
        run_metadata=perception_run_metadata,
    )
    if perception_output_dir
    else None
)

print("==============================================")
print("Test version: multi-agent-eval-v1-headless")
print("Test process PID:", os.getpid())
print("Launchfile:", launchfile)
print("Model file:", file_name)
print("Actor mode:", actor_mode)
if actor_mode == "residual":
    print("Residual hidden dim:", residual_hidden_dim)
    print("Residual scale:", network.residual_scale)
print("Actor selection mode:", actor_selection_mode)
if dual_actor_enabled:
    print("Dual actor mode: enabled")
    print("Standard actor file:", standard_actor_file)
    print("Dense actor file:", dense_actor_file)
    print("Dense actor mode:", dense_actor_mode)
    if actor_selection_mode == "hard_switch":
        print("Switch on distance:", switch_on_distance)
        print("Switch off distance:", switch_off_distance)
        print("Switch on visible neighbors:", switch_on_visible_neighbors)
    elif actor_selection_mode == "case_oracle":
        print("Case oracle map:", case_oracle_map_path)
    elif actor_selection_mode == "interaction_oracle":
        print("Oracle interaction distance:", interaction_oracle_distance)
    elif actor_selection_mode == "recovery_oracle":
        print("Recovery oracle candidate distance:", recovery_oracle_candidate_distance)
        print("Recovery oracle release distance:", recovery_oracle_release_distance)
        print("Recovery oracle progress threshold:", recovery_oracle_progress_threshold)
        print("Recovery oracle progress window:", recovery_oracle_progress_window)
        print(
            "Recovery oracle distance delta threshold:",
            recovery_oracle_distance_delta_threshold,
        )
        print("Recovery oracle goal distance:", recovery_oracle_goal_distance)
        print(
            "Recovery oracle minimum hold steps:",
            recovery_oracle_minimum_hold_steps,
        )
        print(
            "Recovery oracle maximum hold steps:",
            recovery_oracle_maximum_hold_steps,
        )
    elif actor_selection_mode == "learned_gate":
        print("Gate checkpoint:", gate_checkpoint_path)
        print("Gate model id:", dense_policy_controller.model_id)
        print("Gate sequence length:", dense_policy_controller.sequence_length)
        print("Gate detector checkpoint:", gate_detector_checkpoint_path)
        print("Gate switch-on threshold:", dense_policy_controller.switch_on_threshold)
        print("Gate switch-off threshold:", dense_policy_controller.switch_off_threshold)
        print("Gate minimum hold steps:", gate_minimum_hold_steps)
        print("Gate maximum candidates:", gate_max_candidates)
        print("Gate evaluation stride:", gate_evaluation_stride)
    elif actor_selection_mode == "min_lidar_gate":
        print("Min-LiDAR switch-on distance:", min_lidar_switch_on_distance)
        print("Min-LiDAR switch-off distance:", min_lidar_switch_off_distance)
        print("Min-LiDAR minimum hold steps:", min_lidar_minimum_hold_steps)
else:
    print("Dual actor mode: disabled")
print("Rule oracle mode:", rule_oracle_mode or "disabled")
if rule_oracle_controller is not None:
    if rule_oracle_mode == "conflict_pair_yield":
        print("Rule oracle stop distance:", rule_oracle_stop_distance)
        print("Rule oracle release distance:", rule_oracle_release_distance)
        print("Rule oracle max yield steps:", rule_oracle_max_yield_steps)
    else:
        print("Right-hand activation distance:", right_hand_activation_distance)
        print("Right-hand release distance:", right_hand_release_distance)
        print("Right-hand frontal angle:", right_hand_frontal_angle)
        print("Right-hand opposing angle:", right_hand_opposing_angle)
        print("Right-hand minimum closing speed:", right_hand_min_closing_speed)
        print("Right-hand maximum TTC:", right_hand_max_ttc)
        print("Right-hand turn action:", right_hand_turn_action)
        print("Right-hand linear speed cap:", right_hand_linear_speed_cap)
        print("Right-hand max override steps:", right_hand_max_override_steps)
    print("Rule oracle schedule:", rule_oracle_schedule)
print("Fixed physics step size:", fixed_physics_step_size or "disabled")
print("Scenario mode:", scenario_mode)
if scenario_mode == "manifest":
    print("Manifest path:", os.environ.get("DRL_MULTI_MANIFEST_PATH", ""))
    print("Manifest sampling:", os.environ.get("DRL_MULTI_MANIFEST_SAMPLING", "cycle"))
print("Seed:", seed)
print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Agent names:", ", ".join(agent_names))
print("TensorBoard log dir:", log_dir)
print("State path:", state_path)
print("Resume mode:", resume_testing)
print("Starting episode:", episode_num)
print("Starting env steps:", total_env_steps)
print("Starting agent samples:", total_agent_samples)
print("Target test episodes:", target_test_episodes or "unlimited")
print("Trajectory JSONL:", trajectory_path or "disabled")
print("Raw lidar recording:", "enabled" if env.record_raw_lidar else "disabled")
print("Robot-perception shards:", perception_output_dir or "disabled")
if perception_recorder is not None:
    print("Robot-perception split:", perception_split)
    print("Robot-perception frame stride:", perception_frame_stride)
if scenario_mode in ("curriculum", "manifest"):
    print("Case-level stats enabled")
print("==============================================")


def rule_enabled_for_episode(completed_episodes):
    if rule_oracle_controller is None:
        return False
    if rule_oracle_schedule == "all":
        return True
    pair_index, position = divmod(int(completed_episodes), 2)
    rule_position = 1 if pair_index % 2 == 0 else 0
    return position == rule_position

states = env.reset()
episode_case_name = current_case_name(env)
episode_rule_enabled = rule_enabled_for_episode(episode_num)
if perception_recorder is not None:
    perception_recorder.begin_scenario(
        episode_case_name, *current_case_perception_metadata(env)
    )
if dense_policy_controller is not None:
    dense_policy_controller.reset(agent_names)
if rule_oracle_controller is not None:
    rule_oracle_controller.reset(agent_names)
active_mask = [True] * len(agent_names)
episode_done = False
episode_env_steps = 0
episode_agent_samples = 0
episode_rewards = np.zeros(len(agent_names), dtype=np.float32)
episode_success_flags = np.zeros(len(agent_names), dtype=np.int32)
episode_collision_flags = np.zeros(len(agent_names), dtype=np.int32)
episode_final_distances = {name: None for name in agent_names}
episode_start_time = time.time()
episode_dense_action_steps = np.zeros(len(agent_names), dtype=np.int32)
episode_standard_action_steps = np.zeros(len(agent_names), dtype=np.int32)
episode_rule_action_steps = np.zeros(len(agent_names), dtype=np.int32)

while True:
    env_actions = []
    step_active_mask = list(active_mask)
    oracle_flags = None
    if actor_selection_mode == "interaction_oracle":
        oracle_contexts = env.build_neighbor_context(
            [[0.0, 0.0] for _ in agent_names],
            max_neighbors=9,
            include_actions=False,
            active_mask=active_mask,
        )
        oracle_flags = interaction_mask(
            oracle_contexts,
            interaction_oracle_distance,
            feature_dim=5,
        )
    step_actor_states = [np.asarray(state, dtype=float).tolist() for state in states]
    step_actor_poses = {
        name: {
            "x": float(env.last_odom[name].pose.pose.position.x),
            "y": float(env.last_odom[name].pose.pose.position.y),
            "yaw": float(env._get_robot_yaw(name)),
            "timestamp": float(env.last_odom[name].header.stamp.to_sec()),
        }
        for name in agent_names
    }
    step_raw_lidar = (
        {
            name: np.asarray(env.raw_lidar_points[name], dtype=float).round(4).tolist()
            for name in agent_names
        }
        if env.record_raw_lidar and trajectory_path
        else None
    )
    step_actor_modes = {}
    step_gate_probabilities = {}
    step_gate_diagnostics = {}
    if perception_recorder is not None:
        perception_recorder.record_frame(
            env.raw_lidar_points,
            {
                name: [
                    step_actor_poses[name]["x"],
                    step_actor_poses[name]["y"],
                    step_actor_poses[name]["yaw"],
                ]
                for name in agent_names
            },
            step_active_mask,
            agent_names,
            timestamps_by_agent={
                name: step_actor_poses[name]["timestamp"] for name in agent_names
            },
            actor_states_by_agent={
                name: step_actor_states[index]
                for index, name in enumerate(agent_names)
            },
        )

    for idx, state in enumerate(states):
        if not active_mask[idx]:
            env_actions.append([0.0, 0.0])
            continue

        if rule_oracle_controller is not None and episode_rule_enabled:
            active_names = {
                agent_names[index]
                for index, is_active in enumerate(active_mask)
                if is_active
            }
            action, is_yielding = rule_oracle_controller.choose_action(
                env, agent_names[idx], state, active_names
            )
            episode_standard_action_steps[idx] += 1
            if is_yielding:
                episode_rule_action_steps[idx] += 1
        elif actor_selection_mode == "interaction_oracle":
            use_dense_actor = bool(oracle_flags[idx])
            policy = dense_network if use_dense_actor else network
            action = policy.get_action(np.array(state))
            if use_dense_actor:
                episode_dense_action_steps[idx] += 1
            else:
                episode_standard_action_steps[idx] += 1
        elif dense_policy_controller is not None:
            if actor_selection_mode == "learned_gate":
                action, mode, gate_probability, _ = (
                    dense_policy_controller.choose_action(
                        env,
                        agent_names[idx],
                        state,
                        logical_time=(episode_env_steps + 1) * 0.2,
                    )
                )
                step_gate_probabilities[agent_names[idx]] = gate_probability
            else:
                action, mode, _, gate_diagnostics = dense_policy_controller.choose_action(
                    env, agent_names[idx], state
                )
                if gate_diagnostics:
                    step_gate_diagnostics[agent_names[idx]] = gate_diagnostics
            step_actor_modes[agent_names[idx]] = mode
            if mode == "dense":
                episode_dense_action_steps[idx] += 1
            else:
                episode_standard_action_steps[idx] += 1
        else:
            action = network.get_action(np.array(state))
            episode_standard_action_steps[idx] += 1
        env_actions.append([(action[0] + 1) / 2, action[1]])

    next_states, rewards, dones, targets, collisions = env.step(env_actions, active_mask)
    total_env_steps += 1
    step_agents = env.last_step_info["agents"]
    append_trajectory_step(
        {
            "episode": episode_num + 1,
            "case": episode_case_name,
            "step": episode_env_steps + 1,
            "active_before": step_active_mask,
            "actor_states": step_actor_states,
            "actor_poses": step_actor_poses,
            "temporal_lidar": (
                {
                    name: [float(value) for value in env.temporal_lidar_data[name]]
                    for name in agent_names
                }
                if env.temporal_lidar_dim
                else None
            ),
            "raw_lidar_points": step_raw_lidar,
            "actions": [[float(value) for value in action] for action in env_actions],
            "actor_modes": step_actor_modes or None,
            "gate_probabilities": step_gate_probabilities or None,
            "gate_diagnostics": step_gate_diagnostics or None,
            "positions": {
                name: [float(value) for value in env.robot_positions[name]]
                for name in agent_names
            },
            "agents": {
                name: {
                    "target": bool(targets[idx]),
                    "collision": bool(collisions[idx]),
                    "distance": float(step_agents[name]["distance"]),
                    "progress": float(step_agents[name]["progress"]),
                    "min_laser": float(step_agents[name]["min_laser"]),
                    "nearest_robot_distance": (
                        float(step_agents[name]["nearest_robot_distance"])
                        if step_agents[name]["nearest_robot_distance"] is not None
                        else None
                    ),
                    "active_visible_neighbor_count": int(
                        step_agents[name]["active_visible_neighbor_count"]
                    ),
                    "nearest_active_visible_neighbor_distance": (
                        float(
                            step_agents[name][
                                "nearest_active_visible_neighbor_distance"
                            ]
                        )
                        if step_agents[name][
                            "nearest_active_visible_neighbor_distance"
                        ]
                        is not None
                        else None
                    ),
                }
                for idx, name in enumerate(agent_names)
            },
        }
    )

    truncated = episode_env_steps + 1 == max_ep
    for idx in range(len(agent_names)):
        if not active_mask[idx]:
            continue

        episode_rewards[idx] += rewards[idx]
        episode_agent_samples += 1
        total_agent_samples += 1
        success, collision = resolve_terminal_outcome(
            episode_success_flags[idx],
            episode_collision_flags[idx],
            targets[idx],
            collisions[idx],
        )
        episode_success_flags[idx] = int(success)
        episode_collision_flags[idx] = int(collision)
        episode_final_distances[agent_names[idx]] = step_agents[agent_names[idx]][
            "distance"
        ]

        if dones[idx] or truncated:
            active_mask[idx] = False

    states = next_states
    episode_env_steps += 1
    if truncated or not any(active_mask):
        episode_done = True

    if not episode_done:
        continue

    episode_num += 1
    if perception_recorder is not None:
        perception_result = perception_recorder.finish_scenario()
        print(
            "Perception shard | case=%s | written=%s | candidates=%i | "
            "visible_robots=%i | missed_visible_robots=%i | path=%s"
            % (
                episode_case_name,
                perception_result["written"],
                perception_result["candidates"],
                perception_result["visible_robots"],
                perception_result["missed_visible_robots"],
                perception_result["path"],
            )
        )
    elapsed = time.time() - episode_start_time
    steps_per_sec = episode_agent_samples / elapsed if elapsed > 0 else 0.0
    success_rate = float(np.mean(episode_success_flags))
    collision_rate = float(np.mean(episode_collision_flags))
    episode_success_count = int(np.sum(episode_success_flags))
    episode_collision_count = int(np.sum(episode_collision_flags))
    episode_unresolved_count = max(
        len(agent_names) - episode_success_count - episode_collision_count, 0
    )
    unresolved_rate = episode_unresolved_count / len(agent_names)
    full_success = int(np.sum(episode_success_flags) == len(agent_names))
    timeout_episode = int(episode_env_steps >= max_ep)
    mean_reward = float(np.mean(episode_rewards))
    mean_final_distance = float(
        np.mean(
            [
                episode_final_distances[name]
                for name in agent_names
                if episode_final_distances[name] is not None
            ]
        )
    )
    gate_stats = (
        dense_policy_controller.episode_stats()
        if actor_selection_mode in ("learned_gate", "min_lidar_gate")
        else {"switches": 0, "mean_probability": 0.0}
    )
    dense_action_share = float(np.sum(episode_dense_action_steps)) / max(
        float(np.sum(episode_dense_action_steps + episode_standard_action_steps)),
        1.0,
    )
    update_case_stats(
        case_stats,
        episode_case_name,
        episode_success_count,
        episode_collision_count,
        episode_unresolved_count,
        full_success,
        timeout_episode,
        episode_env_steps,
        mean_final_distance,
    )

    success_count += episode_success_count
    collision_count += episode_collision_count
    unresolved_count += episode_unresolved_count
    full_success_count += full_success
    timeout_episode_count += timeout_episode
    if episode_success_count < len(success_hist):
        success_hist[episode_success_count] += 1
    if episode_collision_count < len(collision_hist):
        collision_hist[episode_collision_count] += 1
    recent_rewards.append(mean_reward)
    recent_success_rates.append(success_rate)
    recent_collision_rates.append(collision_rate)
    recent_unresolved_rates.append(unresolved_rate)
    recent_full_success.append(full_success)
    recent_timeout_episodes.append(timeout_episode)

    avg_reward = float(np.mean(recent_rewards[-print_every_episodes:]))
    avg_success = float(np.mean(recent_success_rates[-print_every_episodes:]))
    avg_collision = float(np.mean(recent_collision_rates[-print_every_episodes:]))
    avg_unresolved = float(np.mean(recent_unresolved_rates[-print_every_episodes:]))
    avg_full_success = float(np.mean(recent_full_success[-print_every_episodes:]))
    avg_timeout_episode = float(
        np.mean(recent_timeout_episodes[-print_every_episodes:])
    )

    print(
        "Episode %i complete | case=%s | env_steps=%i | agent_samples=%i | episode_env_steps=%i | "
        "episode_agent_samples=%i | mean_reward=%.3f | success=%i/%i | collision=%i/%i | "
        "unresolved=%i/%i | full_success=%i | timeout=%i | "
        "mean_final_distance=%.3f | dense_action_share=%.3f | gate_switches=%i | "
        "gate_mean_probability=%.3f | rule_enabled=%i | "
        "rule_action_share=%.3f | "
        "samples/sec=%.3f"
        % (
            episode_num,
            episode_case_name,
            total_env_steps,
            total_agent_samples,
            episode_env_steps,
            episode_agent_samples,
            mean_reward,
            episode_success_count,
            len(agent_names),
            episode_collision_count,
            len(agent_names),
            episode_unresolved_count,
            len(agent_names),
            full_success,
            timeout_episode,
            mean_final_distance,
            dense_action_share,
            gate_stats["switches"],
            gate_stats["mean_probability"],
            int(episode_rule_enabled),
            (
                float(np.sum(episode_rule_action_steps))
                / max(float(episode_agent_samples), 1.0)
            ),
            steps_per_sec,
        )
    )

    if episode_num % print_every_episodes == 0:
        print(
            "Recent %i episodes | avg_reward=%.3f | success_rate=%.3f | collision_rate=%.3f | "
            "unresolved_rate=%.3f | full_success_rate=%.3f | timeout_episode_rate=%.3f | "
            "total_success=%i | total_collision=%i | total_unresolved=%i | "
            "total_full_success=%i | timeout_episodes=%i | success_hist=%s | collision_hist=%s"
            % (
                print_every_episodes,
                avg_reward,
                avg_success,
                avg_collision,
                avg_unresolved,
                avg_full_success,
                avg_timeout_episode,
                success_count,
                collision_count,
                unresolved_count,
                full_success_count,
                timeout_episode_count,
                success_hist,
                collision_hist,
            )
        )
        if scenario_mode == "curriculum":
            print_case_stats(case_stats)

    writer.add_scalar("test/episode_mean_reward", mean_reward, episode_num)
    writer.add_scalar("test/episode_success_rate", success_rate, episode_num)
    writer.add_scalar("test/episode_collision_rate", collision_rate, episode_num)
    writer.add_scalar("test/episode_unresolved_rate", unresolved_rate, episode_num)
    writer.add_scalar("test/episode_full_success", full_success, episode_num)
    writer.add_scalar("test/episode_timeout", timeout_episode, episode_num)
    writer.add_scalar("test/mean_final_distance", mean_final_distance, episode_num)
    writer.add_scalar("test/samples_per_sec", steps_per_sec, episode_num)
    writer.add_scalar("test/recent_avg_reward", avg_reward, episode_num)
    writer.add_scalar("test/recent_success_rate", avg_success, episode_num)
    writer.add_scalar("test/recent_collision_rate", avg_collision, episode_num)
    writer.add_scalar("test/recent_unresolved_rate", avg_unresolved, episode_num)
    writer.add_scalar("test/recent_full_success_rate", avg_full_success, episode_num)
    writer.add_scalar(
        "test/recent_timeout_episode_rate", avg_timeout_episode, episode_num
    )
    writer.flush()

    save_test_state(
        {
            "episode_num": episode_num,
            "total_env_steps": total_env_steps,
            "total_agent_samples": total_agent_samples,
            "success_count": success_count,
            "collision_count": collision_count,
            "unresolved_count": unresolved_count,
            "full_success_count": full_success_count,
            "timeout_episode_count": timeout_episode_count,
            "success_hist": success_hist,
            "collision_hist": collision_hist,
            "case_stats": case_stats,
            "manifest_sampling_state": (
                env.manifest_sampling_state() if scenario_mode == "manifest" else None
            ),
            "last_episode_mean_reward": mean_reward,
        }
    )
    append_stats(
        [
            episode_num,
            total_env_steps,
            total_agent_samples,
            episode_env_steps,
            episode_agent_samples,
            mean_reward,
            episode_success_count,
            episode_collision_count,
            full_success,
            mean_final_distance,
            episode_unresolved_count,
            timeout_episode,
            episode_case_name,
            int(episode_rule_enabled),
            dense_action_share,
            gate_stats["switches"],
            gate_stats["mean_probability"],
        ]
    )

    if target_test_episodes and episode_num >= target_test_episodes:
        print(
            "Target test episodes reached | episode_num=%i | target=%i"
            % (episode_num, target_test_episodes)
        )
        writer.close()
        break

    states = env.reset()
    episode_case_name = current_case_name(env)
    episode_rule_enabled = rule_enabled_for_episode(episode_num)
    if perception_recorder is not None:
        perception_recorder.begin_scenario(
            episode_case_name, *current_case_perception_metadata(env)
        )
    if dense_policy_controller is not None:
        dense_policy_controller.reset(agent_names)
    if rule_oracle_controller is not None:
        rule_oracle_controller.reset(agent_names)
    active_mask = [True] * len(agent_names)
    episode_done = False
    episode_env_steps = 0
    episode_agent_samples = 0
    episode_rewards = np.zeros(len(agent_names), dtype=np.float32)
    episode_success_flags = np.zeros(len(agent_names), dtype=np.int32)
    episode_collision_flags = np.zeros(len(agent_names), dtype=np.int32)
    episode_final_distances = {name: None for name in agent_names}
    episode_start_time = time.time()
    episode_dense_action_steps = np.zeros(len(agent_names), dtype=np.int32)
    episode_standard_action_steps = np.zeros(len(agent_names), dtype=np.int32)
    episode_rule_action_steps = np.zeros(len(agent_names), dtype=np.int32)
