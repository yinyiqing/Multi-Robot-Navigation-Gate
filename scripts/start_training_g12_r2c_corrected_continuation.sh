#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
BASE="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线"
TRAIN_MANIFEST="$BASE/datasets/fixed_v1/views/g12_r3_mixed_v1/train.json.gz"
EVAL_MANIFEST="$BASE/datasets/fixed_v1/views/g12_full_scene_selection_v1/validation.json.gz"
LOAD_MODEL="capacity_wide_r2_s4_broad_n5_seed20260816_epoch_001"
MODEL_NAME="capacity_wide_r2c_corrected_n5_seed20260827"
LOG_DIR="$PROJECT_ROOT/logs/active/capacity-wide-g12-r2c-corrected"
PID_FILE="$PROJECT_ROOT/.g12_r2c_corrected.pid"
LAUNCHFILE="$LOG_DIR/runtime_g12_r2c_corrected.launch"
LOCK_FILE=/tmp/local_critic_multi_robot_training.lock
ROS_PORT=15851
GAZEBO_PORT=15951

verify_sha() {
  local path="$1" expected="$2"
  [[ -f "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]] || {
    echo "SHA-256 mismatch: $path" >&2; exit 1
  }
}

verify_sha "$TRAIN_MANIFEST" c2ce37e51e8e98423d6ed6d295a7f5cf54d02e76c42f6459ce35003c899e0841
verify_sha "$EVAL_MANIFEST" 52435d6c5bdf9914e7212dd29cb4bfec074257f72d85f0d71741deee7c63b635
verify_sha "$TD3_DIR/pytorch_models/${LOAD_MODEL}_actor.pth" ace910553931873a275d66e3a964fd2b4716d30b6c68c8dcb3e7af96e56783ee
verify_sha "$TD3_DIR/pytorch_models/${LOAD_MODEL}_critic.pth" eb25c18db9f4b8a272a760f4b7bd5c306f67af34813e9a7ce49e5ded7bb3852c

if [[ "${DRL_MULTI_DRY_RUN:-0}" == 1 ]]; then
  echo "R2C corrected continuation dry run passed"
  echo "Warm start: $LOAD_MODEL (Actor + original 24-dim Critic)"
  echo "Actor: 24-1137-855-2 | budget: 20k | seed: 20260827"
  exit 0
fi

if [[ -f "$PID_FILE" ]]; then
  pid="$(tr -d '[:space:]' < "$PID_FILE")"
  [[ ! "$pid" =~ ^[0-9]+$ ]] || ! kill -0 "$pid" 2>/dev/null || {
    echo "R2C corrected continuation already runs as PID $pid" >&2; exit 1
  }
  unlink "$PID_FILE"
fi
for artifact in "$TD3_DIR/checkpoints/${MODEL_NAME}_latest.pt" "$TD3_DIR/pytorch_models/${MODEL_NAME}_actor.pth"; do
  [[ ! -e "$artifact" ]] || { echo "Output already exists: $artifact" >&2; exit 1; }
done
pgrep -af '^python3(\.8)? -u (train|test)_velodyne_td3_multi.py($| )' >/dev/null && {
  echo "Another multi-robot run is active" >&2; exit 1;
} || true
pgrep -af '(^|/)(gzserver|rosmaster)( |$)' >/dev/null && {
  echo "An existing Gazebo or ROS master is active" >&2; exit 1;
} || true
flock -n "$LOCK_FILE" -c true || { echo "Training lock is busy" >&2; exit 1; }

gpu_free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d '[:space:]')"
gpu_util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 | tr -d '[:space:]')"
(( gpu_free_mib >= 8192 && gpu_util <= 20 )) || {
  echo "GPU 0 is too busy: free=${gpu_free_mib}MiB util=${gpu_util}%" >&2; exit 1;
}

mkdir -p "$LOG_DIR"
/usr/bin/python3 "$PROJECT_ROOT/scripts/generate_multi_robot_launch.py" --num-agents 5 --output "$LAUNCHFILE"
train_log="$LOG_DIR/train_${MODEL_NAME}.log"
runner_log="$LOG_DIR/runner.log"

setsid bash -lc "
  set -eo pipefail
  exec 9>'$LOCK_FILE'
  flock -n 9 || exit 1
  cleanup() {
    pgid=\"\$(ps -o pgid= -p \$\$ | tr -d ' ')\"
    ps -eo pid=,pgid= | awk -v p=\"\$pgid\" -v s=\"\$\$\" '\$2 == p && \$1 != s {print \$1}' | xargs -r kill 2>/dev/null || true
    unlink '$PID_FILE' 2>/dev/null || true
  }
  trap cleanup EXIT
  source /opt/ros/noetic/setup.bash
  source '$PROJECT_ROOT/env.python.sh'
  source '$PROJECT_ROOT/catkin_ws/devel_isolated/setup.bash'
  export CUDA_VISIBLE_DEVICES=0 ROS_HOSTNAME=localhost GAZEBO_IP=127.0.0.1
  export ROS_MASTER_URI=http://localhost:$ROS_PORT ROS_PORT_SIM=$ROS_PORT
  export GAZEBO_MASTER_URI=http://localhost:$GAZEBO_PORT
  export GAZEBO_RESOURCE_PATH='$PROJECT_ROOT/catkin_ws/src/multi_robot_scenario/launch'
  export DRL_MULTI_NUM_AGENTS=5 DRL_MULTI_SEED=20260827
  export DRL_MULTI_TRAIN_LAUNCHFILE='$LAUNCHFILE'
  export DRL_MULTI_SCENARIO=manifest DRL_MULTI_MANIFEST_PATH='$TRAIN_MANIFEST'
  export DRL_MULTI_EVAL_MANIFEST_PATH='$EVAL_MANIFEST' DRL_MULTI_MANIFEST_SAMPLING=cycle
  export DRL_MULTI_TRAIN_FILE_NAME='$MODEL_NAME'
  export DRL_MULTI_TRAINING_VERSION=g12-r2c-corrected-continuation-v1
  export DRL_MULTI_LOAD_MODEL=1 DRL_MULTI_LOAD_MODEL_NAME='$LOAD_MODEL'
  export DRL_MULTI_LOAD_ACTOR_ONLY=0 DRL_MULTI_REQUIRE_MODEL_LOAD=1 DRL_MULTI_RESUME_TRAINING=0
  export DRL_MULTI_ACTOR_HIDDEN_DIM_1=1137 DRL_MULTI_ACTOR_HIDDEN_DIM_2=855 DRL_MULTI_ACTOR_TRAIN_MODE=full
  export DRL_MULTI_USE_LOCAL_CRITIC=0 DRL_MULTI_USE_DYNAMIC_REWARD=0 DRL_MULTI_USE_DISTANCE_WEIGHTED_REWARD=0
  export DRL_MULTI_PROGRESS_REWARD_WEIGHT=20.0 DRL_MULTI_FORWARD_REWARD_WEIGHT=0.5
  export DRL_MULTI_TURN_PENALTY_WEIGHT=0.2 DRL_MULTI_OBSTACLE_PENALTY_WEIGHT=0.5
  export DRL_MULTI_STAGNATION_PENALTY_WEIGHT=0
  export DRL_MULTI_BATCH_SIZE=256 DRL_MULTI_MIN_REPLAY_SIZE=5000 DRL_MULTI_DISCOUNT=0.999 DRL_MULTI_TAU=0.005
  export DRL_MULTI_POLICY_NOISE=0.2 DRL_MULTI_NOISE_CLIP=0.5 DRL_MULTI_POLICY_FREQ=2
  export DRL_MULTI_ACTOR_LR=0.00008 DRL_MULTI_CRITIC_LR=0.00008 DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS=0
  export DRL_MULTI_CRITIC_WARMUP_EXPL_NOISE=0.10 DRL_MULTI_EXPL_NOISE=0.10 DRL_MULTI_EXPL_MIN=0.03
  export DRL_MULTI_EXPL_DECAY_STEPS=20000 DRL_MULTI_RANDOM_LINEAR_EXPLORATION_STEPS=0
  export DRL_MULTI_EVAL_FREQ_AGENT_SAMPLES=10000 DRL_MULTI_EVAL_EPISODES=120 DRL_MULTI_MAX_EPOCHS=2
  export DRL_MULTI_BEST_METRIC=full_success DRL_MULTI_EARLY_STOP_PATIENCE=0
  export DRL_MULTI_FIXED_PHYSICS_STEP_SIZE=0.001 DRL_MULTI_REQUIRE_FIXED_STEP_SERVICE=1
  cd '$TD3_DIR'
  python3 -u train_velodyne_td3_multi.py
" >"$train_log" 2>&1 &
echo $! > "$PID_FILE"
echo "Started R2C corrected continuation"
echo "PID: $(cat "$PID_FILE")"
echo "Log: $train_log"
echo "Warm start: $LOAD_MODEL (Actor + Critic)"
