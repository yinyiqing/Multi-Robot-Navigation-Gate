#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
VIEW_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/g12_r2_curriculum_v1/n3"
TRAIN_MANIFEST="$VIEW_DIR/train.json.gz"
EVAL_MANIFEST="$VIEW_DIR/validation.json.gz"
MODEL_NAME="capacity_wide_r2_s3_broad_n3_seed20260815"
LOAD_MODEL="capacity_wide_r2_s2_broad_n2_seed20260814_best"
PID_FILE="$PROJECT_ROOT/.g12_r2_s3_n3.pid"
LOCK_FILE="/tmp/local_critic_multi_robot_training.lock"
LOG_DIR="$PROJECT_ROOT/logs/active/capacity-wide-g12-r2/s3-n3"
LAUNCHFILE="$LOG_DIR/runtime_g12_r2_s3_n3.launch"
TRAIN_LOG="${1:?training log path is required}"
ROS_PORT=14851
GAZEBO_PORT=14951

stop_runtime_children() {
  local pgid child_pids
  pgid="$(ps -o pgid= -p $$ | tr -d ' ')"
  child_pids="$(ps -eo pid=,pgid= | awk -v pgid="$pgid" -v self="$$" '$2 == pgid && $1 != self { print $1 }')"
  if [[ -n "$child_pids" ]]; then
    xargs -r kill -TERM 2>/dev/null <<<"$child_pids" || true
    sleep 3
    child_pids="$(ps -eo pid=,pgid= | awk -v pgid="$pgid" -v self="$$" '$2 == pgid && $1 != self { print $1 }')"
    [[ -z "$child_pids" ]] || xargs -r kill -KILL 2>/dev/null <<<"$child_pids" || true
  fi
  fuser -k -TERM "${ROS_PORT}/tcp" "${GAZEBO_PORT}/tcp" >/dev/null 2>&1 || true
}

cleanup() {
  stop_runtime_children
  unlink "$PID_FILE" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

exec 9>"$LOCK_FILE"
flock -n 9 || { echo "Multi-robot training lock is busy" >&2; exit 1; }

set +u
source /opt/ros/noetic/setup.bash
source "$PROJECT_ROOT/env.python.sh"
source "$PROJECT_ROOT/catkin_ws/devel_isolated/setup.bash"
set -u

export CUDA_VISIBLE_DEVICES=0
export ROS_HOSTNAME=localhost
export GAZEBO_IP=127.0.0.1
export ROS_MASTER_URI="http://localhost:$ROS_PORT"
export ROS_PORT_SIM="$ROS_PORT"
export GAZEBO_MASTER_URI="http://localhost:$GAZEBO_PORT"
export GAZEBO_RESOURCE_PATH="$PROJECT_ROOT/catkin_ws/src/multi_robot_scenario/launch"

export DRL_MULTI_NUM_AGENTS=3
export DRL_MULTI_SEED=20260815
export DRL_MULTI_TRAIN_LAUNCHFILE="$LAUNCHFILE"
export DRL_MULTI_SCENARIO=manifest
export DRL_MULTI_MANIFEST_PATH="$TRAIN_MANIFEST"
export DRL_MULTI_EVAL_MANIFEST_PATH="$EVAL_MANIFEST"
export DRL_MULTI_MANIFEST_SAMPLING=random
export DRL_MULTI_TRAIN_FILE_NAME="$MODEL_NAME"
export DRL_MULTI_TRAINING_VERSION=g12-r2-s3-wide-n3-pilot-v1
export DRL_MULTI_LOAD_MODEL=1
export DRL_MULTI_LOAD_MODEL_NAME="$LOAD_MODEL"
export DRL_MULTI_LOAD_ACTOR_ONLY=0
export DRL_MULTI_REQUIRE_MODEL_LOAD=1
export DRL_MULTI_RESUME_TRAINING=0

export DRL_MULTI_ACTOR_HIDDEN_DIM_1=1137
export DRL_MULTI_ACTOR_HIDDEN_DIM_2=855
export DRL_MULTI_ACTOR_TRAIN_MODE=full
export DRL_MULTI_USE_LOCAL_CRITIC=0
export DRL_MULTI_USE_DYNAMIC_REWARD=0
export DRL_MULTI_USE_DISTANCE_WEIGHTED_REWARD=0
export DRL_MULTI_USE_LOCAL_NAVIGATION_REWARD=0
export DRL_MULTI_USE_WALL_CLEARANCE_REWARD=0
export DRL_MULTI_PROGRESS_REWARD_WEIGHT=20.0
export DRL_MULTI_FORWARD_REWARD_WEIGHT=0.5
export DRL_MULTI_TURN_PENALTY_WEIGHT=0.2
export DRL_MULTI_OBSTACLE_PENALTY_WEIGHT=0.5
export DRL_MULTI_STAGNATION_PENALTY_WEIGHT=0
unset DRL_MULTI_TIMEOUT_REWARD

export DRL_MULTI_BATCH_SIZE=256
export DRL_MULTI_MIN_REPLAY_SIZE=5000
export DRL_MULTI_DISCOUNT=0.999
export DRL_MULTI_TAU=0.005
export DRL_MULTI_POLICY_NOISE=0.2
export DRL_MULTI_NOISE_CLIP=0.5
export DRL_MULTI_POLICY_FREQ=2
export DRL_MULTI_ACTOR_LR=0.00008
export DRL_MULTI_CRITIC_LR=0.00008
export DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS=0
export DRL_MULTI_ACTOR_ANCHOR_WEIGHT=0
export DRL_MULTI_CRITIC_WARMUP_EXPL_NOISE=0.10
export DRL_MULTI_EXPL_NOISE=0.10
export DRL_MULTI_EXPL_MIN=0.03
export DRL_MULTI_EXPL_DECAY_STEPS=20000
export DRL_MULTI_RANDOM_LINEAR_EXPLORATION_STEPS=0

export DRL_MULTI_EVAL_FREQ_AGENT_SAMPLES=10000
export DRL_MULTI_EVAL_EPISODES=120
export DRL_MULTI_MAX_EPOCHS=2
export DRL_MULTI_BEST_METRIC=full_success
export DRL_MULTI_EARLY_STOP_PATIENCE=0
export DRL_MULTI_FIXED_PHYSICS_STEP_SIZE=0.001
export DRL_MULTI_REQUIRE_FIXED_STEP_SERVICE=1

echo "Starting G12-R2-S3 three-robot pilot"
echo "Model: $MODEL_NAME"
echo "Warm start: $LOAD_MODEL (Actor and Critic)"
echo "Train: $TRAIN_MANIFEST"
echo "Validation: $EVAL_MANIFEST"
echo "Training log: $TRAIN_LOG"
cd "$TD3_DIR"
python3 -u train_velodyne_td3_multi.py >>"$TRAIN_LOG" 2>&1
echo "G12-R2-S3 three-robot pilot complete."

