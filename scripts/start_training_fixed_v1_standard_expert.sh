#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
DATASET_DIR="$PROJECT_ROOT/experiments/04_保留专门化/05_论文主线/datasets/fixed_v1"
LOG_DIR="$PROJECT_ROOT/logs"
PID_FILE="$PROJECT_ROOT/.train_fixed_v1_standard_expert.pid"
NUM_AGENTS=5
MODEL_NAME="${DRL_MULTI_TRAIN_FILE_NAME:-standard_expert_5d_fixed_v1}"
LOAD_MODEL_NAME="${DRL_MULTI_LOAD_MODEL_NAME:-TD3_velodyne_multi_v4_curriculum_stage2_to_5d_geo_critic_from_5a_guarded_best}"
LOAD_ACTOR_ONLY="${DRL_MULTI_LOAD_ACTOR_ONLY:-1}"
ACTOR_ANCHOR_WEIGHT="${DRL_MULTI_ACTOR_ANCHOR_WEIGHT:-0.0}"
ACTOR_UPDATE_DELAY_STEPS="${DRL_MULTI_LOCAL_CRITIC_ACTOR_UPDATE_DELAY_STEPS:-20000}"
TRAINING_VERSION="${DRL_MULTI_TRAINING_VERSION:-standard-expert-fixed-v1}"
ROS_PORT="${DRL_MULTI_TRAIN_ROS_PORT:-11801}"
GAZEBO_PORT="${DRL_MULTI_TRAIN_GAZEBO_PORT:-11901}"
MAX_EPOCHS="${DRL_MULTI_MAX_EPOCHS:-1}"
EVAL_EPISODES="${DRL_MULTI_EVAL_EPISODES:-10}"
LAUNCHFILE="multi_robot_scenario_fixed_v1_standard_train_5.launch"

if [[ ! -f "$TD3_DIR/pytorch_models/${LOAD_MODEL_NAME}_actor.pth" ]]; then
  echo "Actor warm start is missing: $TD3_DIR/pytorch_models/${LOAD_MODEL_NAME}_actor.pth"
  exit 1
fi
if [[ "$LOAD_ACTOR_ONLY" == "0" ]] && [[ ! -f "$TD3_DIR/pytorch_models/${LOAD_MODEL_NAME}_critic.pth" ]]; then
  echo "Critic warm start is missing: $TD3_DIR/pytorch_models/${LOAD_MODEL_NAME}_critic.pth"
  exit 1
fi
if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "standard expert training is already running with PID $old_pid"
    exit 1
  fi
fi

mkdir -p "$LOG_DIR"
python3 "$PROJECT_ROOT/scripts/generate_multi_robot_launch.py" \
  --num-agents "$NUM_AGENTS" \
  --output "$TD3_DIR/assets/$LAUNCHFILE"

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$LOG_DIR/train_fixed_v1_standard_expert_${timestamp}.log"

setsid bash -lc "
  source /opt/ros/noetic/setup.bash
  source '$PROJECT_ROOT/env.python.sh'
  export ROS_HOSTNAME=localhost
  export ROS_MASTER_URI=http://localhost:${ROS_PORT}
  export ROS_PORT_SIM=${ROS_PORT}
  export GAZEBO_MASTER_URI=http://localhost:${GAZEBO_PORT}
  export GAZEBO_RESOURCE_PATH='$PROJECT_ROOT/catkin_ws/src/multi_robot_scenario/launch'
  export DRL_MULTI_NUM_AGENTS=${NUM_AGENTS}
  export DRL_MULTI_TRAIN_LAUNCHFILE='$LAUNCHFILE'
  export DRL_MULTI_SCENARIO=manifest
  export DRL_MULTI_MANIFEST_PATH='$DATASET_DIR/standard/train.json.gz'
  export DRL_MULTI_EVAL_MANIFEST_PATH='$DATASET_DIR/standard/validation.json.gz'
  export DRL_MULTI_MANIFEST_SAMPLING=random
  export DRL_MULTI_TRAIN_FILE_NAME='$MODEL_NAME'
  export DRL_MULTI_LOAD_MODEL=1
  export DRL_MULTI_LOAD_ACTOR_ONLY=${LOAD_ACTOR_ONLY}
  export DRL_MULTI_LOAD_MODEL_NAME='$LOAD_MODEL_NAME'
  export DRL_MULTI_RESUME_TRAINING=1
  export DRL_MULTI_MAX_EPOCHS=${MAX_EPOCHS}
  export DRL_MULTI_EVAL_EPISODES=${EVAL_EPISODES}
  export DRL_MULTI_BEST_METRIC=full_success
  export DRL_MULTI_TRAINING_VERSION='$TRAINING_VERSION'
  export DRL_MULTI_ACTOR_TRAIN_MODE=full
  export DRL_MULTI_USE_DYNAMIC_REWARD=1
  export DRL_MULTI_REWARD_MODE=average
  export DRL_MULTI_REWARD_SELF_WEIGHT=0.8
  export DRL_MULTI_USE_DISTANCE_WEIGHTED_REWARD=1
  export DRL_MULTI_REWARD_SIGMA=2.0
  export DRL_MULTI_USE_LOCAL_CRITIC=1
  export DRL_MULTI_LOCAL_CRITIC_GEOMETRY_ONLY=1
  export DRL_MULTI_LOCAL_CRITIC_MAX_AGENTS=10
  export DRL_MULTI_ACTIVE_NEIGHBORS_ONLY=1
  export DRL_MULTI_USE_LOCAL_NAVIGATION_REWARD=0
  export DRL_MULTI_USE_WALL_CLEARANCE_REWARD=0
  export DRL_MULTI_EXPL_NOISE=0.025
  export DRL_MULTI_EXPL_MIN=0.012
  export DRL_MULTI_ACTOR_LR=0.000001
  export DRL_MULTI_CRITIC_LR=0.00008
  export DRL_MULTI_ACTOR_ANCHOR_WEIGHT=${ACTOR_ANCHOR_WEIGHT}
  export DRL_MULTI_LOCAL_CRITIC_ACTOR_UPDATE_DELAY_STEPS=${ACTOR_UPDATE_DELAY_STEPS}
  cd '$PROJECT_ROOT/catkin_ws'
  source devel_isolated/setup.bash
  cd '$TD3_DIR'
  exec python3 -u train_velodyne_td3_multi.py
" >"$log_file" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "Started standard expert training."
echo "PID: $(cat "$PID_FILE")"
echo "Train manifest: $DATASET_DIR/standard/train.json.gz"
echo "Eval manifest: $DATASET_DIR/standard/validation.json.gz"
echo "Max epochs: $MAX_EPOCHS"
echo "Actor-only warm start: $LOAD_ACTOR_ONLY"
echo "Actor anchor weight: $ACTOR_ANCHOR_WEIGHT"
echo "Actor update delay: $ACTOR_UPDATE_DELAY_STEPS"
echo "Log: $log_file"
