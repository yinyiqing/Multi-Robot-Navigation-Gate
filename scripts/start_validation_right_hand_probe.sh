#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
DATASET_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1"
LOG_DIR="$PROJECT_ROOT/logs"
MANIFEST_PATH="$DATASET_DIR/views/dense_validation_monitor_ultrafast_v3/validation.json.gz"
LAUNCHFILE="multi_robot_scenario_strong_interaction_pilot_5.launch"

MODE="${DRL_MULTI_RIGHT_HAND_PROBE_MODE:-baseline}"
REPEAT="${DRL_MULTI_RIGHT_HAND_PROBE_REPEAT:-1}"
TARGET_EPISODES="${DRL_MULTI_RIGHT_HAND_PROBE_EPISODES:-50}"
SEED="${DRL_MULTI_RIGHT_HAND_PROBE_SEED:-20260731}"
ROS_PORT="${DRL_MULTI_RIGHT_HAND_PROBE_ROS_PORT:-13831}"
GAZEBO_PORT="${DRL_MULTI_RIGHT_HAND_PROBE_GAZEBO_PORT:-13931}"
FIXED_STEP_SIZE="${DRL_MULTI_FIXED_PHYSICS_STEP_SIZE:-0.001}"
ACTOR="${DRL_MULTI_RIGHT_HAND_PROBE_ACTOR:-TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best}"
LOCK_FILE="/tmp/local_critic_multi_robot_fixed_step_validation.lock"
RESUME_RUN_NAME="${DRL_MULTI_RIGHT_HAND_PROBE_RESUME_RUN_NAME:-}"

case "$MODE" in
  baseline)
    RULE_MODE=""
    RULE_SCHEDULE="all"
    MANIFEST_SAMPLING="cycle"
    ;;
  right_hand)
    RULE_MODE="right_hand_pass"
    RULE_SCHEDULE="all"
    MANIFEST_SAMPLING="cycle"
    ;;
  paired)
    RULE_MODE="right_hand_pass"
    RULE_SCHEDULE="paired_alternating"
    MANIFEST_SAMPLING="paired_cycle"
    ;;
  *)
    echo "Probe mode must be baseline, right_hand, or paired"
    exit 2
    ;;
esac

[[ "$REPEAT" =~ ^[1-9][0-9]*$ ]] || { echo "Repeat must be positive"; exit 2; }
[[ "$TARGET_EPISODES" =~ ^[1-9][0-9]*$ ]] || {
  echo "Target episodes must be positive"
  exit 2
}
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "Seed must be an integer"; exit 2; }
[[ "$FIXED_STEP_SIZE" =~ ^0\.[0-9]+$ ]] || {
  echo "Fixed physics step size must be a positive decimal"
  exit 2
}
[[ -f "$MANIFEST_PATH" ]] || { echo "Manifest is missing: $MANIFEST_PATH"; exit 1; }
[[ -f "$TD3_DIR/assets/$LAUNCHFILE" ]] || { echo "Launch file is missing: $LAUNCHFILE"; exit 1; }
[[ -f "$TD3_DIR/pytorch_models/${ACTOR}_actor.pth" ]] || {
  echo "Actor is missing: $TD3_DIR/pytorch_models/${ACTOR}_actor.pth"
  exit 1
}

active_evaluations="$(
  pgrep -af '^python3 -u test_velodyne_td3_multi.py($| )' || true
)"
if [[ -n "$active_evaluations" ]]; then
  echo "Another multi-robot evaluation is running; fixed-step Gazebo must run alone:"
  echo "$active_evaluations"
  exit 1
fi
if ! flock -n "$LOCK_FILE" -c true; then
  echo "Another fixed-step validation holds $LOCK_FILE"
  exit 1
fi

PID_FILE="$PROJECT_ROOT/.right_hand_probe_${MODE}_r${REPEAT}.pid"
if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Probe is already running with PID $old_pid"
    exit 1
  fi
  unlink "$PID_FILE"
fi

for port in "$ROS_PORT" "$GAZEBO_PORT"; do
  if ss -ltnH | awk '{print $4}' | rg -q ":${port}$"; then
    echo "Port $port is already in use"
    exit 1
  fi
done

mkdir -p "$LOG_DIR"
timestamp="$(date +%Y%m%d_%H%M%S)"
run_name="${RESUME_RUN_NAME:-right_hand_probe_${MODE}_r${REPEAT}_s${SEED}_${timestamp}}"
log_file="$LOG_DIR/${run_name}.log"
runner_log="$LOG_DIR/${run_name}_runner.log"
state_path="./checkpoints/${run_name}_state.pt"
stats_path="./results/${run_name}.npy"
if [[ -n "$RESUME_RUN_NAME" ]]; then
  [[ -f "$TD3_DIR/${state_path#./}" ]] || {
    echo "Resume state is missing: $TD3_DIR/${state_path#./}"
    exit 1
  }
  [[ -f "$TD3_DIR/${stats_path#./}" ]] || {
    echo "Resume stats are missing: $TD3_DIR/${stats_path#./}"
    exit 1
  }
fi

setsid bash -lc "
  set -eo pipefail
  exec 9>'$LOCK_FILE'
  flock -n 9 || { echo 'Fixed-step validation lock is busy'; exit 1; }
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
  export DRL_MULTI_SEED=$SEED
  export DRL_MULTI_TEST_LAUNCHFILE='$LAUNCHFILE'
  export DRL_MULTI_TEST_FILE_NAME='$run_name'
  export DRL_MULTI_STANDARD_ACTOR_FILE='$ACTOR'
  export DRL_MULTI_ACTOR_SELECTION_MODE=single
  export DRL_MULTI_TEST_ACTOR_MODE=full
  export DRL_MULTI_TEST_TARGET_EPISODES=$TARGET_EPISODES
  export DRL_MULTI_TEST_STATE_PATH='$state_path'
  export DRL_MULTI_TEST_STATS_PATH='$stats_path'
  export DRL_MULTI_SCENARIO=manifest
  export DRL_MULTI_MANIFEST_PATH='$MANIFEST_PATH'
  export DRL_MULTI_MANIFEST_SAMPLING='$MANIFEST_SAMPLING'
  export DRL_MULTI_FIXED_PHYSICS_STEP_SIZE='$FIXED_STEP_SIZE'
  export DRL_MULTI_RULE_ORACLE_MODE='$RULE_MODE'
  export DRL_MULTI_RULE_ORACLE_SCHEDULE='$RULE_SCHEDULE'
  export DRL_MULTI_RIGHT_HAND_ACTIVATION_DISTANCE=1.5
  export DRL_MULTI_RIGHT_HAND_RELEASE_DISTANCE=1.8
  export DRL_MULTI_RIGHT_HAND_FRONTAL_ANGLE_DEG=35
  export DRL_MULTI_RIGHT_HAND_OPPOSING_ANGLE_DEG=150
  export DRL_MULTI_RIGHT_HAND_MIN_CLOSING_SPEED=0.2
  export DRL_MULTI_RIGHT_HAND_MAX_TTC=3.0
  export DRL_MULTI_RIGHT_HAND_TURN_ACTION=-0.6
  export DRL_MULTI_RIGHT_HAND_LINEAR_SPEED_CAP=0.45
  export DRL_MULTI_RIGHT_HAND_MAX_OVERRIDE_STEPS=20
  unset DRL_MULTI_DENSE_ACTOR_FILE DRL_MULTI_DENSE_ACTOR_MODE
  unset DRL_MULTI_CASE_ORACLE_MAP
  cd '$TD3_DIR'
  python3 -u test_velodyne_td3_multi.py >>'$log_file' 2>&1
" >>"$runner_log" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "Started fixed-step right-hand probe."
echo "PID: $(cat "$PID_FILE")"
echo "Mode: $MODE"
echo "Repeat: $REPEAT"
echo "Episodes: $TARGET_EPISODES"
echo "Actor: $ACTOR"
echo "Fixed physics step: $FIXED_STEP_SIZE"
[[ -n "$RESUME_RUN_NAME" ]] && echo "Resuming run: $RESUME_RUN_NAME"
echo "Log: $log_file"
echo "Stats: $TD3_DIR/${stats_path#./}"
