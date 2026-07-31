#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
DATASET_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1"
LOG_DIR="$PROJECT_ROOT/logs"
MANIFEST_PATH="$DATASET_DIR/views/dense_validation_monitor_v1/validation.json.gz"
LAUNCHFILE="multi_robot_scenario_strong_interaction_pilot_5.launch"

ACTOR="${DRL_MULTI_VALIDATION_ACTOR:-TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best}"
LABEL="${DRL_MULTI_VALIDATION_LABEL:-5a}"
REPEAT="${DRL_MULTI_VALIDATION_REPEAT:-1}"
SEED="${DRL_MULTI_VALIDATION_SEED:-20260731}"
ROS_PORT="${DRL_MULTI_VALIDATION_ROS_PORT:-13821}"
GAZEBO_PORT="${DRL_MULTI_VALIDATION_GAZEBO_PORT:-13921}"
TARGET_EPISODES=200
SAFE_LABEL="${LABEL//[^A-Za-z0-9_]/_}"
PID_FILE="$PROJECT_ROOT/.validation200_dense_${SAFE_LABEL}_r${REPEAT}.pid"

[[ "$REPEAT" =~ ^[1-9][0-9]*$ ]] || { echo "Repeat must be positive"; exit 2; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "Seed must be an integer"; exit 2; }
[[ -f "$MANIFEST_PATH" ]] || { echo "Manifest is missing: $MANIFEST_PATH"; exit 1; }
[[ -f "$TD3_DIR/assets/$LAUNCHFILE" ]] || { echo "Launch file is missing: $LAUNCHFILE"; exit 1; }
[[ -f "$TD3_DIR/pytorch_models/${ACTOR}_actor.pth" ]] || {
  echo "Actor is missing: $TD3_DIR/pytorch_models/${ACTOR}_actor.pth"
  exit 1
}

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Validation is already running with PID $old_pid"
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
run_name="validation200_dense_${SAFE_LABEL}_r${REPEAT}_s${SEED}_${timestamp}"
log_file="$LOG_DIR/${run_name}.log"
runner_log="$LOG_DIR/${run_name}_runner.log"
state_path="./checkpoints/${run_name}_state.pt"
stats_path="./results/${run_name}.npy"

setsid bash -lc "
  set -eo pipefail
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
  export DRL_MULTI_MANIFEST_SAMPLING=cycle
  unset DRL_MULTI_DENSE_ACTOR_FILE DRL_MULTI_DENSE_ACTOR_MODE
  unset DRL_MULTI_CASE_ORACLE_MAP DRL_MULTI_RULE_ORACLE_MODE
  cd '$TD3_DIR'
  python3 -u test_velodyne_td3_multi.py >'$log_file' 2>&1
" >"$runner_log" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "Started fixed 200-scene standalone Actor validation."
echo "PID: $(cat "$PID_FILE")"
echo "Actor: $ACTOR"
echo "Label: $LABEL"
echo "Repeat: $REPEAT"
echo "Seed: $SEED"
echo "Manifest: $MANIFEST_PATH"
echo "Log: $log_file"
echo "Stats: $TD3_DIR/${stats_path#./}"
