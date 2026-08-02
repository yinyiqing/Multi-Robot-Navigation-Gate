#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
DATASET_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1"
TRAIN_MANIFEST="$DATASET_DIR/dense/train.json.gz"
EVAL_MANIFEST="$DATASET_DIR/views/dense_validation_monitor_ultrafast_v3/validation.json.gz"
SOURCE_MODEL="interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726"
WEAK_MODEL="TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best"
MODEL_NAME="${DRL_MULTI_TRAIN_FILE_NAME:-actor_b_from_epoch16_full_pilot_v1_s20260802}"
SAFE_MODEL="${MODEL_NAME//[^A-Za-z0-9_]/_}"
SOURCE_CHECKPOINT="$TD3_DIR/checkpoints/${SOURCE_MODEL}_latest.pt"
CHECKPOINT="$TD3_DIR/checkpoints/${MODEL_NAME}_latest.pt"
PID_FILE="${DRL_MULTI_PID_FILE:-$PROJECT_ROOT/.train_${SAFE_MODEL}.pid}"
LOCK_FILE="/tmp/local_critic_multi_robot_training.lock"
LOG_DIR="$PROJECT_ROOT/logs"
LAUNCHFILE="multi_robot_scenario_strong_interaction_pilot_5.launch"
ROS_PORT="${DRL_MULTI_ROS_PORT:-14201}"
GAZEBO_PORT="${DRL_MULTI_GAZEBO_PORT:-14301}"
SEED="${DRL_MULTI_SEED:-20260802}"
RESUME_TRAINING="${DRL_MULTI_RESUME_TRAINING:-0}"
MAX_EPOCHS="${DRL_MULTI_MAX_EPOCHS:-18}"
ACTOR_UPDATE_DELAY="${DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS:-326500}"

for path in \
  "$TRAIN_MANIFEST" \
  "$EVAL_MANIFEST" \
  "$SOURCE_CHECKPOINT" \
  "$TD3_DIR/pytorch_models/${WEAK_MODEL}_actor.pth" \
  "$TD3_DIR/assets/$LAUNCHFILE"; do
  [[ -f "$path" ]] || { echo "Required file is missing: $path"; exit 1; }
done
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "DRL_MULTI_SEED must be an integer"; exit 2; }
[[ "$MAX_EPOCHS" =~ ^[1-9][0-9]*$ ]] || {
  echo "DRL_MULTI_MAX_EPOCHS must be positive"
  exit 2
}
[[ "$ACTOR_UPDATE_DELAY" =~ ^[0-9]+$ ]] || {
  echo "DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS must be non-negative"
  exit 2
}
[[ "$RESUME_TRAINING" == 0 || "$RESUME_TRAINING" == 1 ]] || {
  echo "DRL_MULTI_RESUME_TRAINING must be 0 or 1"
  exit 2
}

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Actor B pilot is already running with PID $old_pid"
    exit 1
  fi
  unlink "$PID_FILE"
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

if [[ "$RESUME_TRAINING" == 0 ]]; then
  [[ ! -e "$CHECKPOINT" ]] || {
    echo "Fresh Actor B checkpoint already exists: $CHECKPOINT"
    exit 1
  }
  cp --reflink=auto "$SOURCE_CHECKPOINT" "$CHECKPOINT"
else
  [[ -f "$CHECKPOINT" ]] || {
    echo "Actor B resume checkpoint is missing: $CHECKPOINT"
    exit 1
  }
fi

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
    ps -eo pid=,pgid= | awk -v pgid=\"\$pgid\" -v self=\"\$\$\" \
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
  unset DRL_MULTI_FIXED_PHYSICS_STEP_SIZE

  export DRL_MULTI_TRAIN_FILE_NAME='$MODEL_NAME'
  export DRL_MULTI_TRAINING_VERSION='actor-b-from-epoch16-full-pilot-v1'
  export DRL_MULTI_RESUME_TRAINING=1
  export DRL_MULTI_MAX_EPOCHS='$MAX_EPOCHS'
  export DRL_MULTI_EVAL_EPISODES=50
  export DRL_MULTI_EVAL_FREQ_AGENT_SAMPLES=5000
  export DRL_MULTI_BEST_METRIC=full_success
  export DRL_MULTI_EARLY_STOP_PATIENCE=1
  export DRL_MULTI_EARLY_STOP_MIN_EPOCHS=18
  export DRL_MULTI_EARLY_STOP_FULL_SUCCESS_DROP=0.10
  export DRL_MULTI_EARLY_STOP_SUCCESS_DROP=0.08
  export DRL_MULTI_EARLY_STOP_TIMEOUT_INCREASE=0.15
  export DRL_MULTI_EARLY_STOP_TIMEOUT_ABSOLUTE=0.30

  export DRL_MULTI_ACTOR_TRAIN_MODE=full
  export DRL_MULTI_USE_ORACLE_INTERACTION_ROLLOUT=0
  export DRL_MULTI_USE_ORACLE_TARGET_POLICY=1
  export DRL_MULTI_ORACLE_WEAK_ACTOR_NAME='$WEAK_MODEL'
  export DRL_MULTI_ORACLE_INTERACTION_DISTANCE=2.0
  export DRL_MULTI_ACTOR_INTERACTION_ONLY=0

  export DRL_MULTI_USE_DYNAMIC_REWARD=1
  export DRL_MULTI_REWARD_MODE=average
  export DRL_MULTI_REWARD_SELF_WEIGHT=0.8
  export DRL_MULTI_USE_DISTANCE_WEIGHTED_REWARD=1
  export DRL_MULTI_REWARD_SIGMA=2.0
  export DRL_MULTI_ACTIVE_NEIGHBORS_ONLY=1
  export DRL_MULTI_FORWARD_REWARD_WEIGHT=0.0
  export DRL_MULTI_STAGNATION_PENALTY_WEIGHT=0.0
  unset DRL_MULTI_TIMEOUT_REWARD

  export DRL_MULTI_USE_LOCAL_CRITIC=1
  export DRL_MULTI_LOCAL_CRITIC_GEOMETRY_ONLY=0
  export DRL_MULTI_LOCAL_CRITIC_CONTEXT_MODE=ego_motion
  export DRL_MULTI_LOCAL_CRITIC_MAX_AGENTS=10
  export DRL_MULTI_CRITIC_INTERACTION_FRACTION=0.5
  export DRL_MULTI_ROBOT_SAFE_DISTANCE=1.2
  export DRL_MULTI_ROBOT_PROXIMITY_PENALTY_WEIGHT=5.0
  export DRL_MULTI_ROBOT_PROXIMITY_SPEED_PENALTY_WEIGHT=10.0
  export DRL_MULTI_ROBOT_CLEARANCE_REWARD_WEIGHT=20.0
  export DRL_MULTI_ROBOT_CLEARANCE_REWARD_MAX_GAIN=0.1

  export DRL_MULTI_USE_ACTOR_GRADIENT_GATE=1
  export DRL_MULTI_ACTOR_GRADIENT_SAFETY_DISTANCE=1.2
  export DRL_MULTI_ACTOR_GRADIENT_GATE_BATCH_SIZE=512
  export DRL_MULTI_ACTOR_GRADIENT_GATE_MIN_SAMPLES=32
  export DRL_MULTI_ACTOR_GRADIENT_MAX_LINEAR_POSITIVE_SHARE=0.9
  export DRL_MULTI_ACTOR_GRADIENT_MAX_ANGULAR_ONE_SIDED_SHARE=0.9
  export DRL_MULTI_CRITIC_SAFETY_RANKING_WEIGHT=5.0
  export DRL_MULTI_CRITIC_SAFETY_RANKING_DISTANCE=1.0
  export DRL_MULTI_CRITIC_SAFETY_RANKING_MIN_CLOSING_SPEED=0.1
  export DRL_MULTI_CRITIC_SAFETY_RANKING_LINEAR_DELTA=0.4
  export DRL_MULTI_CRITIC_SAFETY_RANKING_MARGIN=0.1
  export DRL_MULTI_ACTOR_SAFETY_FOCUSED=0
  export DRL_MULTI_ACTOR_ANGULAR_ANCHOR_WEIGHT=0.0
  export DRL_MULTI_ACTOR_ANCHOR_WEIGHT=0.0
  export DRL_MULTI_ACTOR_Q_NORMALIZATION_ALPHA=0.0

  export DRL_MULTI_EXPL_NOISE=0.03
  export DRL_MULTI_CRITIC_WARMUP_EXPL_NOISE=0.03
  export DRL_MULTI_EXPL_MIN=0.03
  export DRL_MULTI_EXPL_DECAY_STEPS=80000
  export DRL_MULTI_ACTOR_LR=0.000001
  export DRL_MULTI_CRITIC_LR=0.00008
  export DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS='$ACTOR_UPDATE_DELAY'
  export DRL_MULTI_POLICY_FREQ=2

  cd '$TD3_DIR'
  python3 -u train_velodyne_td3_multi.py >>'$log_file' 2>&1
" >>"$runner_log" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "Started Actor B full-navigation pilot."
echo "PID: $(cat "$PID_FILE")"
echo "Model: $MODEL_NAME"
echo "Forked checkpoint: $SOURCE_CHECKPOINT"
echo "Execution: Actor B controls every state; no Oracle rollout"
echo "Target policy: frozen 5A normally, Actor B within 2.0 m (training only)"
echo "Epoch 17: Actor remains frozen through the first matched 50-scene baseline"
echo "Epoch 18: Actor unlocks at sample $ACTOR_UPDATE_DELAY for the first all-state interval"
echo "Log: $log_file"
echo "Runner log: $runner_log"
