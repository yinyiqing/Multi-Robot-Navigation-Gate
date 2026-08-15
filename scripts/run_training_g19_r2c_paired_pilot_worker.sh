#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$ROOT/TD3"
BASE="experiments/03_保留专门化/02_论文主线"
TRAIN_MANIFEST="$ROOT/$BASE/datasets/fixed_v1/views/g12_r3_mixed_v1/train.json.gz"
EVAL_MANIFEST="$ROOT/$BASE/datasets/fixed_v1/views/g12_full_scene_selection_v1/validation.json.gz"
SOURCE_MODEL="TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best"
LOG_DIR="$ROOT/logs/active/g19-r2c-paired-pilot"
ARCHIVE_DIR="$ROOT/logs/archive/training/g19_r2c_paired_pilot"
LAUNCHFILE="$LOG_DIR/runtime_g19_r2c.launch"
PID_FILE="$ROOT/.g19_r2c_paired_pilot.pid"
LOCK_FILE=/tmp/local_critic_multi_robot_training.lock
ROS_PORT=17623
GAZEBO_PORT=17723

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
export DRL_MULTI_NUM_AGENTS=5 DRL_MULTI_SEED=20260826
export DRL_MULTI_TRAIN_LAUNCHFILE="$LAUNCHFILE"
export DRL_MULTI_SCENARIO=manifest DRL_MULTI_MANIFEST_PATH="$TRAIN_MANIFEST"
export DRL_MULTI_EVAL_MANIFEST_PATH="$EVAL_MANIFEST" DRL_MULTI_MANIFEST_SAMPLING=cycle
export DRL_MULTI_LOAD_MODEL=1 DRL_MULTI_LOAD_ACTOR_ONLY=1 DRL_MULTI_REQUIRE_MODEL_LOAD=1
export DRL_MULTI_LOAD_MODEL_NAME="$SOURCE_MODEL" DRL_MULTI_RESUME_TRAINING=0
export DRL_MULTI_ACTOR_TRAIN_MODE=full

export DRL_MULTI_USE_LOCAL_CRITIC=1 DRL_MULTI_LOCAL_CRITIC_GEOMETRY_ONLY=0
export DRL_MULTI_LOCAL_CRITIC_CONTEXT_MODE=ego_motion DRL_MULTI_LOCAL_CRITIC_MAX_AGENTS=10
export DRL_MULTI_ACTIVE_NEIGHBORS_ONLY=1 DRL_MULTI_CRITIC_INTERACTION_FRACTION=0.50
export DRL_MULTI_USE_ORACLE_INTERACTION_ROLLOUT=0 DRL_MULTI_USE_ORACLE_TARGET_POLICY=0
export DRL_MULTI_ACTOR_INTERACTION_ONLY=0 DRL_MULTI_USE_ACTOR_GRADIENT_GATE=0
export DRL_MULTI_ACTOR_SAFETY_FOCUSED=0

export DRL_MULTI_USE_DYNAMIC_REWARD=1 DRL_MULTI_REWARD_MODE=average
export DRL_MULTI_REWARD_SELF_WEIGHT=0.8 DRL_MULTI_USE_DISTANCE_WEIGHTED_REWARD=1
export DRL_MULTI_REWARD_SIGMA=2.0 DRL_MULTI_INTERACTION_SAFE_DISTANCE=1.2
export DRL_MULTI_INTERACTION_CLOSE_PENALTY=0.5 DRL_MULTI_INTERACTION_STAGNATION_PENALTY=0.05
export DRL_MULTI_ROBOT_SAFE_DISTANCE=1.2 DRL_MULTI_ROBOT_PROXIMITY_PENALTY_WEIGHT=5.0
export DRL_MULTI_ROBOT_PROXIMITY_SPEED_PENALTY_WEIGHT=10.0
export DRL_MULTI_ROBOT_CLEARANCE_REWARD_WEIGHT=20.0 DRL_MULTI_ROBOT_CLEARANCE_REWARD_MAX_GAIN=0.1
export DRL_MULTI_USE_SAFE_RECOVERY_REWARD=0 DRL_MULTI_USE_ANTI_STAGNATION_REWARD=0
export DRL_MULTI_USE_LOCAL_NAVIGATION_REWARD=0 DRL_MULTI_USE_WALL_CLEARANCE_REWARD=0
export DRL_MULTI_USE_YIELD_PRIORITY_REWARD=0 DRL_MULTI_FORWARD_REWARD_WEIGHT=0.0
export DRL_MULTI_STAGNATION_PENALTY_WEIGHT=0.0

export DRL_MULTI_BATCH_SIZE=256 DRL_MULTI_MIN_REPLAY_SIZE=5000
export DRL_MULTI_DISCOUNT=0.99999 DRL_MULTI_TAU=0.005
export DRL_MULTI_POLICY_NOISE=0.2 DRL_MULTI_NOISE_CLIP=0.5 DRL_MULTI_POLICY_FREQ=2
export DRL_MULTI_ACTOR_LR=0.000002 DRL_MULTI_CRITIC_LR=0.00002
export DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS=41000
export DRL_MULTI_ACTOR_Q_NORMALIZATION_ALPHA=1.0 DRL_MULTI_ACTOR_ANCHOR_WEIGHT=1.0
export DRL_MULTI_ACTOR_ANCHOR_SAFE_ONLY=1 DRL_MULTI_ACTOR_ANCHOR_SAFE_DISTANCE=2.0
export DRL_MULTI_ACTOR_GRAD_NORM_CLIP=1.0
export DRL_MULTI_CRITIC_WARMUP_EXPL_NOISE=0.08
export DRL_MULTI_EXPL_NOISE=0.025 DRL_MULTI_EXPL_MIN=0.012
export DRL_MULTI_EXPL_DECAY_STEPS=120000 DRL_MULTI_RANDOM_LINEAR_EXPLORATION_STEPS=0
export DRL_MULTI_EVAL_FREQ_AGENT_SAMPLES=20000 DRL_MULTI_EVAL_EPISODES=120
export DRL_MULTI_MAX_EPOCHS=3 DRL_MULTI_BEST_METRIC=full_success
export DRL_MULTI_EARLY_STOP_PATIENCE=0
export DRL_MULTI_FIXED_PHYSICS_STEP_SIZE=0.001 DRL_MULTI_REQUIRE_FIXED_STEP_SERVICE=1

run_one() {
  local kind="$1" hidden1="$2" hidden2="$3" expansion="$4"
  local model="capacity_${kind}_g19_r2c_n5_seed20260826"
  local log="$LOG_DIR/train_${model}.log"
  export DRL_MULTI_TRAIN_FILE_NAME="$model"
  export DRL_MULTI_TRAINING_VERSION="g19-r2c-${kind}-paired-stability-v1"
  export DRL_MULTI_ACTOR_HIDDEN_DIM_1="$hidden1" DRL_MULTI_ACTOR_HIDDEN_DIM_2="$hidden2"
  export DRL_MULTI_ALLOW_ACTOR_WARMSTART_EXPANSION="$expansion"
  echo "Starting G19-R2C $kind"
  echo "Model: $model | Actor: 24-$hidden1-$hidden2-2 | budget: 60k"
  (cd "$TD3_DIR" && python3 -u train_velodyne_td3_multi.py) >"$log" 2>&1
  stop_runtime
  echo "Completed G19-R2C $kind"
}

mkdir -p "$LOG_DIR"
run_one original 800 600 0
python3 "$ROOT/scripts/analyze_g19_r2c_paired_pilot.py" --require-control-pass \
  >"$LOG_DIR/original_gate.json"
run_one wide 1137 855 1
python3 "$ROOT/scripts/analyze_g19_r2c_paired_pilot.py" >"$LOG_DIR/summary.json"
[[ ! -e "$ARCHIVE_DIR" ]] || { echo "Archive already exists: $ARCHIVE_DIR" >&2; exit 1; }
mkdir -p "$(dirname "$ARCHIVE_DIR")"
mv "$LOG_DIR" "$ARCHIVE_DIR"
