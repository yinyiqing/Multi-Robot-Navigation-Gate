#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
VIEW_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/g12_r2_curriculum_v1/n1"
TRAIN_MANIFEST="$VIEW_DIR/train.json.gz"
EVAL_MANIFEST="$VIEW_DIR/validation.json.gz"
MODEL_NAME="${DRL_MULTI_TRAIN_FILE_NAME:-current_generalist_n1_original_broad_s20260810}"
SAFE_MODEL="${MODEL_NAME//[^A-Za-z0-9_]/_}"
PID_FILE="${DRL_MULTI_PID_FILE:-$PROJECT_ROOT/.train_${SAFE_MODEL}.pid}"
LOG_DIR="${DRL_MULTI_LOG_DIR:-$PROJECT_ROOT/logs/active/current-generalist-r2style/n1}"
LOCK_FILE="/tmp/local_critic_multi_robot_training.lock"
LAUNCHFILE="$LOG_DIR/runtime_current_generalist_n1.launch"
ROS_PORT="${DRL_MULTI_ROS_PORT:-14811}"
GAZEBO_PORT="${DRL_MULTI_GAZEBO_PORT:-14911}"
SEED="${DRL_MULTI_SEED:-20260811}"

declare -A EXPECTED_SHA256=(
  ["$TRAIN_MANIFEST"]="c71e4e87bbc528782cb76dc7df076c493900523bb748b3fb646f3d77fa5f0263"
  ["$EVAL_MANIFEST"]="9ab4c5913f683d01e3ab186ea591d373abe1e835180f4a0bfeb469990269b125"
)
for path in "${!EXPECTED_SHA256[@]}"; do
  [[ -f "$path" ]] || { echo "Required N1 input is missing: $path" >&2; exit 1; }
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "${EXPECTED_SHA256[$path]}" ]] || {
    echo "N1 input hash mismatch: $path" >&2
    echo "expected=${EXPECTED_SHA256[$path]} actual=$actual" >&2
    exit 1
  }
done

if [[ "${DRL_MULTI_DRY_RUN:-0}" == 1 ]]; then
  cat <<EOF
Current-generalist N1 dry run passed.
Model: $MODEL_NAME
Initialization: random original-width Actor and random original 24-dim Critic
Actor: 24->800->600->2
Agents: 1
Budget: 5 x 20000 = 100000 agent samples
Train: $TRAIN_MANIFEST
Validation: $EVAL_MANIFEST
Logs: $LOG_DIR
EOF
  exit 0
fi

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Current-generalist N1 is already running with PID $old_pid" >&2
    exit 1
  fi
  unlink "$PID_FILE"
fi
if [[ -e "$TD3_DIR/checkpoints/${MODEL_NAME}_latest.pt" ]]; then
  echo "Fresh-run checkpoint already exists: $TD3_DIR/checkpoints/${MODEL_NAME}_latest.pt" >&2
  exit 1
fi
if pgrep -af '^python3(\.8)? -u (train|test)_velodyne_td3_multi.py($| )' >/dev/null; then
  echo "Another multi-robot training or evaluation process is running" >&2
  exit 1
fi
if pgrep -af '(^|/)(gzserver|rosmaster)( |$)' >/dev/null; then
  echo "An existing Gazebo or ROS master is running; N1 will not start a second simulator" >&2
  exit 1
fi
if ! flock -n "$LOCK_FILE" -c true; then
  echo "Another multi-robot process holds $LOCK_FILE" >&2
  exit 1
fi
for port in "$ROS_PORT" "$GAZEBO_PORT"; do
  if ss -ltnH | awk '{print $4}' | grep -Eq ":${port}$"; then
    echo "Port $port is already in use" >&2
    exit 1
  fi
done

gpu_free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d '[:space:]')"
gpu_util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 | tr -d '[:space:]')"
if (( gpu_free_mib < 4096 || gpu_util > 35 )); then
  echo "GPU 0 is not available enough: free=${gpu_free_mib}MiB util=${gpu_util}%" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
/usr/bin/python3 "$PROJECT_ROOT/scripts/generate_multi_robot_launch.py" \
  --num-agents 1 --output "$LAUNCHFILE"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$LOG_DIR/train_${SAFE_MODEL}_${timestamp}.log"
runner_log="$LOG_DIR/train_${SAFE_MODEL}_${timestamp}_runner.log"

setsid bash -lc "
  set -eo pipefail
  exec 9>'$LOCK_FILE'
  flock -n 9 || { echo 'Multi-robot training lock is busy'; exit 1; }
  cleanup() {
    pgid=\"\$(ps -o pgid= -p \$\$ | tr -d ' ')\"
    ps -eo pid=,pgid= | awk -v pgid=\"\$pgid\" -v self=\"\$\$\" \\
      '\$2 == pgid && \$1 != self { print \$1 }' | xargs -r kill 2>/dev/null || true
    unlink '$PID_FILE' 2>/dev/null || true
  }
  trap cleanup EXIT
  source /opt/ros/noetic/setup.bash
  source '$PROJECT_ROOT/env.python.sh'
  source '$PROJECT_ROOT/catkin_ws/devel_isolated/setup.bash'
  export CUDA_VISIBLE_DEVICES=0
  export ROS_HOSTNAME=localhost
  export GAZEBO_IP=127.0.0.1
  export ROS_MASTER_URI=http://localhost:$ROS_PORT
  export ROS_PORT_SIM=$ROS_PORT
  export GAZEBO_MASTER_URI=http://localhost:$GAZEBO_PORT
  export GAZEBO_RESOURCE_PATH='$PROJECT_ROOT/catkin_ws/src/multi_robot_scenario/launch'

  export DRL_MULTI_NUM_AGENTS=1
  export DRL_MULTI_SEED='$SEED'
  export DRL_MULTI_TRAIN_LAUNCHFILE='$LAUNCHFILE'
  export DRL_MULTI_SCENARIO=manifest
  export DRL_MULTI_MANIFEST_PATH='$TRAIN_MANIFEST'
  export DRL_MULTI_EVAL_MANIFEST_PATH='$EVAL_MANIFEST'
  export DRL_MULTI_MANIFEST_SAMPLING=random
  export DRL_MULTI_TRAIN_FILE_NAME='$MODEL_NAME'
  export DRL_MULTI_TRAINING_VERSION=current-generalist-r2style-n1-original-random-v1
  export DRL_MULTI_LOAD_MODEL=0
  export DRL_MULTI_LOAD_ACTOR_ONLY=0
  export DRL_MULTI_REQUIRE_MODEL_LOAD=0
  export DRL_MULTI_RESUME_TRAINING=0

  export DRL_MULTI_ACTOR_HIDDEN_DIM_1=800
  export DRL_MULTI_ACTOR_HIDDEN_DIM_2=600
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
  export DRL_MULTI_USE_SAFE_RECOVERY_REWARD=0
  export DRL_MULTI_USE_ANTI_STAGNATION_REWARD=0
  export DRL_MULTI_ROBOT_SAFE_DISTANCE=0.0
  export DRL_MULTI_ROBOT_PROXIMITY_PENALTY_WEIGHT=5.0
  export DRL_MULTI_ROBOT_PROXIMITY_SPEED_PENALTY_WEIGHT=0
  export DRL_MULTI_ROBOT_CLEARANCE_REWARD_WEIGHT=0
  export DRL_MULTI_USE_YIELD_PRIORITY_REWARD=0

  export DRL_MULTI_BATCH_SIZE=256
  export DRL_MULTI_MIN_REPLAY_SIZE=5000
  export DRL_MULTI_DISCOUNT=0.999
  export DRL_MULTI_TAU=0.005
  export DRL_MULTI_POLICY_NOISE=0.2
  export DRL_MULTI_NOISE_CLIP=0.5
  export DRL_MULTI_POLICY_FREQ=2
  export DRL_MULTI_ACTOR_LR=0.0001
  export DRL_MULTI_CRITIC_LR=0.0001
  export DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS=0
  export DRL_MULTI_CRITIC_WARMUP_EXPL_NOISE=0.35
  export DRL_MULTI_EXPL_NOISE=0.35
  export DRL_MULTI_EXPL_MIN=0.08
  export DRL_MULTI_EXPL_DECAY_STEPS=100000
  export DRL_MULTI_RANDOM_LINEAR_EXPLORATION_STEPS=5000
  export DRL_MULTI_RANDOM_LINEAR_EXPLORATION_SCOPE=all

  export DRL_MULTI_EVAL_FREQ_AGENT_SAMPLES=20000
  export DRL_MULTI_EVAL_EPISODES=120
  export DRL_MULTI_MAX_EPOCHS=5
  export DRL_MULTI_BEST_METRIC=full_success
  export DRL_MULTI_EARLY_STOP_PATIENCE=0
  unset DRL_MULTI_FIXED_PHYSICS_STEP_SIZE
  unset DRL_MULTI_REQUIRE_FIXED_STEP_SERVICE
  unset DRL_MULTI_TIMEOUT_REWARD

  cd '$TD3_DIR'
  python3 -u train_velodyne_td3_multi.py >>'$log_file' 2>&1
" >>"$runner_log" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "Started current-generalist N1."
echo "PID: $(cat "$PID_FILE")"
echo "Model: $MODEL_NAME"
echo "Initialization: random original-width Actor and Critic"
echo "Actor: 24->800->600->2"
echo "GPU: 0 (free=${gpu_free_mib}MiB util=${gpu_util}%)"
echo "Budget: 5 x 20000 = 100000 agent samples; eval=120"
echo "Log: $log_file"
echo "Runner log: $runner_log"
