#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
VIEW_DIR="$PROJECT_ROOT/experiments/04_保留专门化/05_论文主线/datasets/fixed_v1/views/weak_interaction_validation_v1"
LOG_DIR="$PROJECT_ROOT/logs"
PID_FILE="$PROJECT_ROOT/.validation_weak_interaction_5a.pid"
MANIFEST_PATH="$VIEW_DIR/validation.json.gz"
LAUNCHFILE="multi_robot_scenario_strong_interaction_pilot_5.launch"
ACTOR="TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best"
ROS_PORT=13810
GAZEBO_PORT=13910
TARGET_EPISODES=248
SEED=20260719

[[ -f "$MANIFEST_PATH" ]] || { echo "Weak-interaction manifest is missing: $MANIFEST_PATH"; exit 1; }
[[ -f "$TD3_DIR/assets/$LAUNCHFILE" ]] || { echo "Launch file is missing: $LAUNCHFILE"; exit 1; }
[[ -f "$TD3_DIR/pytorch_models/${ACTOR}_actor.pth" ]] || { echo "5A actor is missing."; exit 1; }
if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "5A weak-interaction validation is already running with PID $old_pid"
    exit 1
  fi
fi
if ss -ltnH | awk '{print $4}' | rg -q ":(${ROS_PORT}|${GAZEBO_PORT})$"; then
  echo "ROS or Gazebo port is already in use."
  exit 1
fi

mkdir -p "$LOG_DIR"
timestamp="$(date +%Y%m%d_%H%M%S)"
run_name="validation_weak_interaction_5a_${timestamp}"
eval_log="$LOG_DIR/${run_name}.log"
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
  export DRL_MULTI_TEST_TARGET_EPISODES=$TARGET_EPISODES
  export DRL_MULTI_TEST_STATE_PATH='$state_path'
  export DRL_MULTI_TEST_STATS_PATH='$stats_path'
  export DRL_MULTI_SCENARIO=manifest
  export DRL_MULTI_MANIFEST_PATH='$MANIFEST_PATH'
  export DRL_MULTI_MANIFEST_SAMPLING=cycle
  unset DRL_MULTI_DENSE_ACTOR_FILE DRL_MULTI_RULE_ORACLE_MODE
  cd '$TD3_DIR'
  python3 -u test_velodyne_td3_multi.py
" >"$eval_log" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "Started 5A weak-interaction validation."
echo "PID: $(cat "$PID_FILE")"
echo "Episodes: $TARGET_EPISODES"
echo "Log: $eval_log"
echo "Expected runtime: roughly 20-30 minutes."
