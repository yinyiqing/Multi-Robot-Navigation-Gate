#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
LOG_DIR="$PROJECT_ROOT/logs"
PID_FILE="$PROJECT_ROOT/.train_multi_stage3_light_patch_detached.pid"
NUM_AGENTS=3
LAUNCHFILE="multi_robot_scenario_curriculum_stage3_light_patch_${NUM_AGENTS}.launch"
LAUNCH_PATH="$TD3_DIR/assets/$LAUNCHFILE"
MODEL_NAME="${DRL_MULTI_TRAIN_FILE_NAME:-TD3_velodyne_multi_v4_curriculum_stage3_light_patch_from_stage2_3d2}"
LOAD_MODEL_NAME="${DRL_MULTI_LOAD_MODEL_NAME:-TD3_velodyne_multi_v4_curriculum_stage2_to_3d2_geo_critic_from_3a_guarded_best}"
CASES_PATH="$PROJECT_ROOT/experiments/多智能体/课程学习/cases/stage3_light_patch_cases.json"
ROS_PORT="${DRL_MULTI_ROS_PORT:-11381}"
GAZEBO_PORT="${DRL_MULTI_GAZEBO_PORT:-11481}"

mkdir -p "$LOG_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$LOG_DIR/train_multi_stage3_light_patch_detached_${timestamp}.log"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "A detached stage3 light-patch training process is already running with PID $old_pid"
    exit 1
  fi
fi

existing_pid="$(
  (
    pgrep -af "^python3(\\.8)? .*train_velodyne_td3_multi\\.py$" \
      | awk 'NR==1 {print $1}'
  ) || true
)"
if [[ -n "$existing_pid" ]]; then
  echo "A multi-agent training process is already running with PID $existing_pid"
  echo "Please stop the current multi-agent training before starting this run."
  exit 1
fi

python3 "$PROJECT_ROOT/scripts/generate_multi_robot_launch.py" \
  --num-agents "$NUM_AGENTS" \
  --output "$LAUNCH_PATH"

setsid bash -lc "
  source /opt/ros/noetic/setup.bash
  source '$PROJECT_ROOT/env.python.sh'
  export ROS_HOSTNAME=localhost
  export ROS_MASTER_URI=http://localhost:${ROS_PORT}
  export ROS_PORT_SIM=${ROS_PORT}
  export GAZEBO_MASTER_URI=http://localhost:${GAZEBO_PORT}
  export GAZEBO_RESOURCE_PATH='$PROJECT_ROOT/catkin_ws/src/multi_robot_scenario/launch'
  export DRL_MULTI_NUM_AGENTS='$NUM_AGENTS'
  export DRL_MULTI_TRAIN_LAUNCHFILE='$LAUNCHFILE'
  export DRL_MULTI_SCENARIO=curriculum
  export DRL_MULTI_CURRICULUM_CASES='$CASES_PATH'
  export DRL_MULTI_CURRICULUM_SAMPLING='${DRL_MULTI_CURRICULUM_SAMPLING:-cycle}'
  export DRL_MULTI_USE_DYNAMIC_REWARD=1
  export DRL_MULTI_REWARD_MODE=average
  export DRL_MULTI_REWARD_SELF_WEIGHT='${DRL_MULTI_REWARD_SELF_WEIGHT:-0.8}'
  export DRL_MULTI_USE_DISTANCE_WEIGHTED_REWARD=1
  export DRL_MULTI_REWARD_SIGMA='${DRL_MULTI_REWARD_SIGMA:-2.0}'
  export DRL_MULTI_INTERACTION_SAFE_DISTANCE='${DRL_MULTI_INTERACTION_SAFE_DISTANCE:-0.9}'
  export DRL_MULTI_INTERACTION_CLOSE_PENALTY='${DRL_MULTI_INTERACTION_CLOSE_PENALTY:-0.35}'
  export DRL_MULTI_INTERACTION_STAGNATION_PENALTY='${DRL_MULTI_INTERACTION_STAGNATION_PENALTY:-0.02}'
  export DRL_MULTI_USE_LOCAL_CRITIC=1
  export DRL_MULTI_LOCAL_CRITIC_GEOMETRY_ONLY=1
  export DRL_MULTI_LOCAL_CRITIC_MAX_AGENTS=10
  export DRL_MULTI_ACTIVE_NEIGHBORS_ONLY=1
  export DRL_MULTI_USE_LOCAL_NAVIGATION_REWARD=1
  export DRL_MULTI_LOCAL_NAV_HEADING_WEIGHT='${DRL_MULTI_LOCAL_NAV_HEADING_WEIGHT:-0.35}'
  export DRL_MULTI_LOCAL_NAV_WRONG_WAY_PENALTY='${DRL_MULTI_LOCAL_NAV_WRONG_WAY_PENALTY:-0.25}'
  export DRL_MULTI_LOCAL_NAV_TURN_WEIGHT='${DRL_MULTI_LOCAL_NAV_TURN_WEIGHT:-0.22}'
  export DRL_MULTI_LOCAL_NAV_NEAR_GOAL_DISTANCE='${DRL_MULTI_LOCAL_NAV_NEAR_GOAL_DISTANCE:-0.9}'
  export DRL_MULTI_LOCAL_NAV_HEADING_ERROR='${DRL_MULTI_LOCAL_NAV_HEADING_ERROR:-0.5}'
  export DRL_MULTI_USE_WALL_CLEARANCE_REWARD=0
  export DRL_MULTI_BEST_METRIC=full_success
  export DRL_MULTI_EVAL_EPISODES='${DRL_MULTI_EVAL_EPISODES:-36}'
  export DRL_MULTI_MAX_EPOCHS='${DRL_MULTI_MAX_EPOCHS:-8}'
  export DRL_MULTI_EXPL_NOISE='${DRL_MULTI_EXPL_NOISE:-0.025}'
  export DRL_MULTI_EXPL_MIN='${DRL_MULTI_EXPL_MIN:-0.012}'
  export DRL_MULTI_ACTOR_LR='${DRL_MULTI_ACTOR_LR:-0.000001}'
  export DRL_MULTI_CRITIC_LR='${DRL_MULTI_CRITIC_LR:-0.00002}'
  export DRL_MULTI_LOCAL_CRITIC_ACTOR_UPDATE_DELAY_STEPS='${DRL_MULTI_LOCAL_CRITIC_ACTOR_UPDATE_DELAY_STEPS:-12000}'
  export DRL_MULTI_TRAINING_VERSION='stage3-light-patch-from-stage2-3d2-v1'
  export DRL_MULTI_TRAIN_FILE_NAME='$MODEL_NAME'
  export DRL_MULTI_LOAD_MODEL=1
  export DRL_MULTI_LOAD_FULL_MODEL_WITH_LOCAL_CRITIC=1
  export DRL_MULTI_LOAD_MODEL_NAME='$LOAD_MODEL_NAME'
  cd '$PROJECT_ROOT/catkin_ws'
  source devel_isolated/setup.bash
  cd '$TD3_DIR'
  exec python3 -u train_velodyne_td3_multi.py
" >"$log_file" 2>&1 < /dev/null &

echo $! > "$PID_FILE"

echo "Detached stage3 light-patch training started."
echo "PID: $(cat "$PID_FILE")"
echo "Agents: $NUM_AGENTS"
echo "Model: $MODEL_NAME"
echo "Warm start: $LOAD_MODEL_NAME"
echo "Warm start mode: actor + geometry local critic"
echo "Cases: $CASES_PATH"
echo "Launch: $LAUNCH_PATH"
echo "Actor update delay steps: ${DRL_MULTI_LOCAL_CRITIC_ACTOR_UPDATE_DELAY_STEPS:-12000}"
echo "Actor LR: ${DRL_MULTI_ACTOR_LR:-0.000001}"
echo "Critic LR: ${DRL_MULTI_CRITIC_LR:-0.00002}"
echo "Exploration noise: ${DRL_MULTI_EXPL_NOISE:-0.025}"
echo "Exploration min: ${DRL_MULTI_EXPL_MIN:-0.012}"
echo "Max epochs: ${DRL_MULTI_MAX_EPOCHS:-8}"
echo "Eval episodes: ${DRL_MULTI_EVAL_EPISODES:-36}"
echo "Log: $log_file"
