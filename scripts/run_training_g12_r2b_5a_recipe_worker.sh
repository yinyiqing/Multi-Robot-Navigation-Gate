#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$ROOT/TD3"
MODEL_NAME="capacity_wide_r2b_5a_recipe_n5_seed20260823"
SOURCE_MODEL="TD3_velodyne_multi_v4_curriculum_stage2_to_3d2_geo_critic_from_3a_guarded_best"
EVAL_MANIFEST="$ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/g12_r2_curriculum_v1/n5/validation.json.gz"
LOG_DIR="$ROOT/logs/active/capacity-wide-g12-r2b-5a-recipe"
LAUNCHFILE="$LOG_DIR/runtime_g12_r2b_5a_recipe_n5.launch"
PID_FILE="$ROOT/.g12_r2b_5a_recipe.pid"
LOCK_FILE=/tmp/local_critic_multi_robot_training.lock
TRAIN_LOG="${1:?training log path is required}"
ROS_PORT=17023
GAZEBO_PORT=17123

stop_runtime() {
  local pgid children
  pgid="$(ps -o pgid= -p $$ | tr -d ' ')"
  children="$(ps -eo pid=,pgid= | awk -v p="$pgid" -v s="$$" '$2 == p && $1 != s {print $1}')"
  [[ -z "$children" ]] || xargs -r kill -TERM 2>/dev/null <<<"$children" || true
  sleep 3
  children="$(ps -eo pid=,pgid= | awk -v p="$pgid" -v s="$$" '$2 == p && $1 != s {print $1}')"
  [[ -z "$children" ]] || xargs -r kill -KILL 2>/dev/null <<<"$children" || true
  fuser -k -KILL "${ROS_PORT}/tcp" "${GAZEBO_PORT}/tcp" >/dev/null 2>&1 || true
}
cleanup() { stop_runtime; unlink "$PID_FILE" 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

exec 9>"$LOCK_FILE"
flock -n 9 || { echo "Multi-robot training lock is busy" >&2; exit 1; }
set +u
source /opt/ros/noetic/setup.bash
source "$ROOT/env.python.sh"
source "$ROOT/catkin_ws/devel_isolated/setup.bash"
set -u

export CUDA_VISIBLE_DEVICES=0
export ROS_HOSTNAME=localhost GAZEBO_IP=127.0.0.1
export ROS_MASTER_URI="http://localhost:$ROS_PORT" ROS_PORT_SIM="$ROS_PORT"
export GAZEBO_MASTER_URI="http://localhost:$GAZEBO_PORT"
export GAZEBO_RESOURCE_PATH="$ROOT/catkin_ws/src/multi_robot_scenario/launch"
export DRL_MULTI_NUM_AGENTS=5 DRL_MULTI_SEED=20260823
export DRL_MULTI_TRAIN_LAUNCHFILE="$LAUNCHFILE"
export DRL_MULTI_SCENARIO=standard
export DRL_MULTI_EVAL_MANIFEST_PATH="$EVAL_MANIFEST"
export DRL_MULTI_TRAIN_FILE_NAME="$MODEL_NAME"
export DRL_MULTI_TRAINING_VERSION=g12-r2b-wide-5a-recipe-n5-v1
export DRL_MULTI_LOAD_MODEL=1 DRL_MULTI_LOAD_ACTOR_ONLY=1 DRL_MULTI_REQUIRE_MODEL_LOAD=1
export DRL_MULTI_LOAD_MODEL_NAME="$SOURCE_MODEL" DRL_MULTI_RESUME_TRAINING=0
export DRL_MULTI_ACTOR_HIDDEN_DIM_1=1137 DRL_MULTI_ACTOR_HIDDEN_DIM_2=855
export DRL_MULTI_ALLOW_ACTOR_WARMSTART_EXPANSION=1 DRL_MULTI_ACTOR_TRAIN_MODE=full

export DRL_MULTI_USE_DYNAMIC_REWARD=0 DRL_MULTI_USE_DISTANCE_WEIGHTED_REWARD=0
export DRL_MULTI_USE_LOCAL_CRITIC=0 DRL_MULTI_ACTIVE_NEIGHBORS_ONLY=1
export DRL_MULTI_USE_LOCAL_NAVIGATION_REWARD=0 DRL_MULTI_USE_WALL_CLEARANCE_REWARD=0
export DRL_MULTI_BATCH_SIZE=40 DRL_MULTI_MIN_REPLAY_SIZE=0
export DRL_MULTI_DISCOUNT=0.99999 DRL_MULTI_TAU=0.005
export DRL_MULTI_POLICY_NOISE=0.2 DRL_MULTI_NOISE_CLIP=0.5 DRL_MULTI_POLICY_FREQ=2
export DRL_MULTI_ACTOR_LR=0.000002 DRL_MULTI_CRITIC_LR=0.00002
export DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS=20000
export DRL_MULTI_EXPL_NOISE=0.025 DRL_MULTI_EXPL_MIN=0.012 DRL_MULTI_EXPL_DECAY_STEPS=500000
export DRL_MULTI_RANDOM_LINEAR_EXPLORATION_STEPS=0
export DRL_MULTI_EVAL_FREQ_AGENT_SAMPLES=10000 DRL_MULTI_EVAL_EPISODES=120
export DRL_MULTI_MAX_EPOCHS=3 DRL_MULTI_BEST_METRIC=full_success
export DRL_MULTI_EARLY_STOP_PATIENCE=0

echo "Starting G12-R2B 5A-recipe wide Actor"
echo "Model: $MODEL_NAME"
echo "Source Actor: $SOURCE_MODEL (function-preserving expansion)"
echo "Validation: $EVAL_MANIFEST"
echo "Budget: 3 x 10k = 30k agent samples"
cd "$TD3_DIR"
python3 -u train_velodyne_td3_multi.py >>"$TRAIN_LOG" 2>&1
echo "G12-R2B training complete"
