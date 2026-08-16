#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$ROOT/TD3"
MODEL_NAME="capacity_wide_r2b_short_cont_n5_seed20260816"
LOAD_MODEL="capacity_wide_r2b_5a_recipe_n5_seed20260823_best"
EVAL_MANIFEST="$ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/g12_r2_curriculum_v1/n5/validation.json.gz"
LOG_DIR="$ROOT/logs/active/capacity-wide-g12-r2b-short-continuation"
RUN_DIR="$ROOT/experiments/03_保留专门化/02_论文主线/12_参数匹配单Actor容量对照/local_data/r2b_short_continuation"
LAUNCHFILE="$LOG_DIR/runtime_g12_r2b_short_continuation.launch"
PID_FILE="$ROOT/.g12_r2b_short_continuation.pid"
LOCK_FILE=/tmp/local_critic_multi_robot_training.lock
ROS_PORT=17823
GAZEBO_PORT=17923

[[ -f "$TD3_DIR/checkpoints/${LOAD_MODEL}.pt" ]] || { echo "Missing R2B epoch1 checkpoint" >&2; exit 1; }
[[ -f "$EVAL_MANIFEST" ]] || { echo "Missing R2B validation manifest" >&2; exit 1; }
[[ ! -e "$TD3_DIR/checkpoints/${MODEL_NAME}_latest.pt" ]] || { echo "Output already exists" >&2; exit 1; }
if [[ -f "$PID_FILE" ]]; then
  pid="$(tr -d '[:space:]' < "$PID_FILE")"
  [[ ! "$pid" =~ ^[0-9]+$ ]] || ! kill -0 "$pid" 2>/dev/null || { echo "Already running" >&2; exit 1; }
  unlink "$PID_FILE"
fi
if pgrep -af '^python3(\.8)? -u (train|test)_velodyne_td3_multi.py($| )' >/dev/null; then
  echo "Another multi-robot run is active" >&2; exit 1
fi
flock -n "$LOCK_FILE" -c true || { echo "Multi-robot lock is busy" >&2; exit 1; }
gpu_free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d '[:space:]')"
gpu_util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 | tr -d '[:space:]')"
(( gpu_free_mib >= 8192 && gpu_util <= 20 )) || { echo "GPU 0 is busy" >&2; exit 1; }

mkdir -p "$LOG_DIR" "$RUN_DIR"
/usr/bin/python3 "$ROOT/scripts/generate_multi_robot_launch.py" --num-agents 5 --output "$LAUNCHFILE"
train_log="$LOG_DIR/train_${MODEL_NAME}.log"

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
  source '$ROOT/env.python.sh'
  source '$ROOT/catkin_ws/devel_isolated/setup.bash'
  export CUDA_VISIBLE_DEVICES=0 ROS_HOSTNAME=localhost GAZEBO_IP=127.0.0.1
  export ROS_MASTER_URI=http://localhost:$ROS_PORT ROS_PORT_SIM=$ROS_PORT GAZEBO_MASTER_URI=http://localhost:$GAZEBO_PORT
  export GAZEBO_RESOURCE_PATH='$ROOT/catkin_ws/src/multi_robot_scenario/launch'
  export DRL_MULTI_NUM_AGENTS=5 DRL_MULTI_SEED=20260816 DRL_MULTI_TRAIN_LAUNCHFILE='$LAUNCHFILE'
  export DRL_MULTI_SCENARIO=standard DRL_MULTI_EVAL_MANIFEST_PATH='$EVAL_MANIFEST' DRL_MULTI_MANIFEST_SAMPLING=cycle
  export DRL_MULTI_TRAIN_FILE_NAME='$MODEL_NAME' DRL_MULTI_TRAINING_VERSION=g12-r2b-short-continuation-v1
  export DRL_MULTI_LOAD_MODEL=1 DRL_MULTI_LOAD_MODEL_NAME='$LOAD_MODEL' DRL_MULTI_LOAD_ACTOR_ONLY=0 DRL_MULTI_REQUIRE_MODEL_LOAD=1 DRL_MULTI_RESUME_TRAINING=0
  export DRL_MULTI_ACTOR_HIDDEN_DIM_1=1137 DRL_MULTI_ACTOR_HIDDEN_DIM_2=855 DRL_MULTI_ACTOR_TRAIN_MODE=full
  export DRL_MULTI_USE_LOCAL_CRITIC=0 DRL_MULTI_USE_DYNAMIC_REWARD=0 DRL_MULTI_USE_DISTANCE_WEIGHTED_REWARD=0
  export DRL_MULTI_PROGRESS_REWARD_WEIGHT=20.0 DRL_MULTI_FORWARD_REWARD_WEIGHT=0.5 DRL_MULTI_TURN_PENALTY_WEIGHT=0.2 DRL_MULTI_OBSTACLE_PENALTY_WEIGHT=0.5
  export DRL_MULTI_STAGNATION_PENALTY_WEIGHT=0
  export DRL_MULTI_BATCH_SIZE=40 DRL_MULTI_MIN_REPLAY_SIZE=0 DRL_MULTI_DISCOUNT=0.99999 DRL_MULTI_TAU=0.005
  export DRL_MULTI_POLICY_NOISE=0.2 DRL_MULTI_NOISE_CLIP=0.5 DRL_MULTI_POLICY_FREQ=2
  export DRL_MULTI_ACTOR_LR=0.000002 DRL_MULTI_CRITIC_LR=0.00002 DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS=0
  export DRL_MULTI_EXPL_NOISE=0.025 DRL_MULTI_EXPL_MIN=0.012 DRL_MULTI_EXPL_DECAY_STEPS=500000 DRL_MULTI_RANDOM_LINEAR_EXPLORATION_STEPS=0
  export DRL_MULTI_EVAL_FREQ_AGENT_SAMPLES=5000 DRL_MULTI_EVAL_EPISODES=120 DRL_MULTI_MAX_EPOCHS=1 DRL_MULTI_BEST_METRIC=full_success DRL_MULTI_EARLY_STOP_PATIENCE=0
  export DRL_MULTI_FIXED_PHYSICS_STEP_SIZE=0.001 DRL_MULTI_REQUIRE_FIXED_STEP_SERVICE=1
  cd '$TD3_DIR'
  python3 -u train_velodyne_td3_multi.py
" >"$train_log" 2>&1 &
echo $! > "$PID_FILE"
echo "Started short R2B continuation"
echo "PID: $(cat "$PID_FILE")"
echo "Log: $train_log"
