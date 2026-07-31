#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
DATASET_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1"
TRAIN_MANIFEST="$DATASET_DIR/dense/train.json.gz"
EVAL_MANIFEST="$DATASET_DIR/views/dense_validation_monitor_ultrafast_v3/validation.json.gz"
BASE_MODEL="TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best"
MODEL_NAME="${DRL_MULTI_TRAIN_FILE_NAME:-independent_dense_actor_simple_td3_hparam_a_s20260801}"
SAFE_MODEL="${MODEL_NAME//[^A-Za-z0-9_]/_}"
PID_FILE="${DRL_MULTI_PID_FILE:-$PROJECT_ROOT/.train_${SAFE_MODEL}.pid}"
LOG_DIR="$PROJECT_ROOT/logs"
LOCK_FILE="/tmp/local_critic_multi_robot_training.lock"
LAUNCHFILE="multi_robot_scenario_strong_interaction_pilot_5.launch"
ROS_PORT="${DRL_MULTI_ROS_PORT:-13801}"
GAZEBO_PORT="${DRL_MULTI_GAZEBO_PORT:-13901}"
SEED="${DRL_MULTI_SEED:-20260801}"
MAX_EPOCHS="${DRL_MULTI_MAX_EPOCHS:-3}"

for path in "$TRAIN_MANIFEST" "$EVAL_MANIFEST"; do
  [[ -f "$path" ]] || { echo "Required manifest is missing: $path"; exit 1; }
done
for path in \
  "$TD3_DIR/pytorch_models/${BASE_MODEL}_actor.pth" \
  "$TD3_DIR/assets/$LAUNCHFILE"; do
  [[ -f "$path" ]] || { echo "Required file is missing: $path"; exit 1; }
done
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "DRL_MULTI_SEED must be an integer"; exit 2; }
[[ "$MAX_EPOCHS" =~ ^[1-9][0-9]*$ ]] || {
  echo "DRL_MULTI_MAX_EPOCHS must be positive"
  exit 2
}

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Simple TD3 experiment A is already running with PID $old_pid"
    exit 1
  fi
  unlink "$PID_FILE"
fi
if [[ -e "$TD3_DIR/checkpoints/${MODEL_NAME}_latest.pt" ]]; then
  echo "Fresh-run checkpoint already exists: $TD3_DIR/checkpoints/${MODEL_NAME}_latest.pt"
  exit 1
fi
if pgrep -af '^python3(\.8)? -u (train|test)_velodyne_td3_multi.py($| )' >/dev/null; then
  echo "Another multi-robot training or evaluation process is running"
  exit 1
fi
if ! flock -n "$LOCK_FILE" -c true; then
  echo "Another multi-robot process holds $LOCK_FILE"
  exit 1
fi
for port in "$ROS_PORT" "$GAZEBO_PORT"; do
  if ss -ltnH | awk '{print $4}' | grep -Eq ":${port}$"; then
    echo "Port $port is already in use"
    exit 1
  fi
done

mkdir -p "$LOG_DIR"
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
  export ROS_HOSTNAME=localhost
  export ROS_MASTER_URI=http://localhost:$ROS_PORT
  export ROS_PORT_SIM=$ROS_PORT
  export GAZEBO_MASTER_URI=http://localhost:$GAZEBO_PORT
  export GAZEBO_RESOURCE_PATH='$PROJECT_ROOT/catkin_ws/src/multi_robot_scenario/launch'

  export DRL_MULTI_NUM_AGENTS=5
  export DRL_MULTI_SEED='$SEED'
  export DRL_MULTI_TRAIN_LAUNCHFILE='$LAUNCHFILE'
  export DRL_MULTI_SCENARIO=manifest
  export DRL_MULTI_MANIFEST_PATH='$TRAIN_MANIFEST'
  export DRL_MULTI_EVAL_MANIFEST_PATH='$EVAL_MANIFEST'
  export DRL_MULTI_MANIFEST_SAMPLING=cycle
  export DRL_MULTI_FIXED_PHYSICS_STEP_SIZE=0.001

  export DRL_MULTI_TRAIN_FILE_NAME='$MODEL_NAME'
  export DRL_MULTI_TRAINING_VERSION='dense-simple-td3-hparam-a-v1'
  export DRL_MULTI_LOAD_MODEL=1
  export DRL_MULTI_LOAD_ACTOR_ONLY=1
  export DRL_MULTI_REQUIRE_MODEL_LOAD=1
  export DRL_MULTI_LOAD_MODEL_NAME='$BASE_MODEL'
  export DRL_MULTI_RESUME_TRAINING=0
  export DRL_MULTI_MAX_EPOCHS='$MAX_EPOCHS'
  export DRL_MULTI_EVAL_EPISODES=50
  export DRL_MULTI_EVAL_FREQ_AGENT_SAMPLES=5000
  export DRL_MULTI_BEST_METRIC=full_success
  export DRL_MULTI_EARLY_STOP_PATIENCE=0

  export DRL_MULTI_ACTOR_TRAIN_MODE=full
  export DRL_MULTI_USE_LOCAL_CRITIC=0
  export DRL_MULTI_USE_ORACLE_INTERACTION_ROLLOUT=0
  export DRL_MULTI_ACTOR_INTERACTION_ONLY=0
  export DRL_MULTI_USE_ACTOR_GRADIENT_GATE=0
  export DRL_MULTI_CRITIC_SAFETY_RANKING_WEIGHT=0
  export DRL_MULTI_ACTOR_SAFETY_FOCUSED=0
  export DRL_MULTI_ACTOR_SLOWDOWN_SAFETY_WEIGHT=0
  export DRL_MULTI_ACTOR_ANGULAR_ANCHOR_WEIGHT=0
  export DRL_MULTI_ACTOR_REFERENCE_ACCELERATION_CAP_WEIGHT=0
  export DRL_MULTI_ACTOR_ANCHOR_WEIGHT=0
  export DRL_MULTI_ACTOR_Q_NORMALIZATION_ALPHA=0

  export DRL_MULTI_USE_DYNAMIC_REWARD=1
  export DRL_MULTI_REWARD_MODE=average
  export DRL_MULTI_REWARD_SELF_WEIGHT=0.8
  export DRL_MULTI_USE_DISTANCE_WEIGHTED_REWARD=1
  export DRL_MULTI_REWARD_SIGMA=2.0
  export DRL_MULTI_PROGRESS_REWARD_WEIGHT=20.0
  export DRL_MULTI_FORWARD_REWARD_WEIGHT=0.5
  export DRL_MULTI_TURN_PENALTY_WEIGHT=0.2
  export DRL_MULTI_OBSTACLE_PENALTY_WEIGHT=0.5
  export DRL_MULTI_STAGNATION_PENALTY_WEIGHT=0
  export DRL_MULTI_ROBOT_SAFE_DISTANCE=0
  export DRL_MULTI_ROBOT_PROXIMITY_PENALTY_WEIGHT=0
  export DRL_MULTI_ROBOT_PROXIMITY_SPEED_PENALTY_WEIGHT=0
  export DRL_MULTI_ROBOT_CLEARANCE_REWARD_WEIGHT=0
  export DRL_MULTI_USE_YIELD_PRIORITY_REWARD=0
  export DRL_MULTI_USE_ANTI_STAGNATION_REWARD=0
  export DRL_MULTI_USE_SAFE_RECOVERY_REWARD=0
  export DRL_MULTI_USE_LOCAL_NAVIGATION_REWARD=0
  export DRL_MULTI_USE_WALL_CLEARANCE_REWARD=0

  export DRL_MULTI_BATCH_SIZE=256
  export DRL_MULTI_MIN_REPLAY_SIZE=6000
  export DRL_MULTI_DISCOUNT=0.995
  export DRL_MULTI_TAU=0.005
  export DRL_MULTI_POLICY_NOISE=0.2
  export DRL_MULTI_NOISE_CLIP=0.5
  export DRL_MULTI_POLICY_FREQ=2
  export DRL_MULTI_ACTOR_LR=0.00001
  export DRL_MULTI_CRITIC_LR=0.0001
  export DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS=0
  export DRL_MULTI_EXPL_NOISE=0.10
  export DRL_MULTI_CRITIC_WARMUP_EXPL_NOISE=0.10
  export DRL_MULTI_EXPL_MIN=0.03
  export DRL_MULTI_EXPL_DECAY_STEPS=100000

  cd '$TD3_DIR'
  python3 -u train_velodyne_td3_multi.py >>'$log_file' 2>&1
" >>"$runner_log" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "Started Dense simple TD3 experiment A."
echo "PID: $(cat "$PID_FILE")"
echo "Model: $MODEL_NAME"
echo "Warm start: 5A Actor; fresh original 24-dimensional Critic"
echo "Updates: Actor and Critic both start after 6000 replay transitions"
echo "Reward: original base terms with 0.8 self + 0.2 neighbors; no add-on rewards"
echo "Budget: $MAX_EPOCHS x 5000 agent samples"
echo "Log: $log_file"
echo "Runner log: $runner_log"
