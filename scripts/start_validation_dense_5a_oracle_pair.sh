#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
DATASET_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1"
LOG_DIR="$PROJECT_ROOT/logs"
MANIFEST_PATH="$DATASET_DIR/dense/validation.json.gz"
LAUNCHFILE="multi_robot_scenario_fixed_v1_dense_validation_5d_5.launch"
BASE_MODEL="TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best"
STRONG_MODEL="interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016"
TARGET_EPISODES=1000
SEED=20260728

[[ -f "$MANIFEST_PATH" ]] || { echo "Dense validation manifest is missing: $MANIFEST_PATH"; exit 1; }
[[ -f "$TD3_DIR/assets/$LAUNCHFILE" ]] || { echo "Launch file is missing: $LAUNCHFILE"; exit 1; }
for model in "$BASE_MODEL" "$STRONG_MODEL"; do
  [[ -f "$TD3_DIR/pytorch_models/${model}_actor.pth" ]] || {
    echo "Actor is missing: $TD3_DIR/pytorch_models/${model}_actor.pth"
    exit 1
  }
done

labels=("5a" "5a_oracle_strong_e16")
selection_modes=("single" "interaction_oracle")
ros_ports=(13601 13602)
gazebo_ports=(13701 13702)

for idx in 0 1; do
  label="${labels[$idx]}"
  pid_file="$PROJECT_ROOT/.validation_dense_${label}.pid"
  if [[ -f "$pid_file" ]]; then
    old_pid="$(tr -d '[:space:]' < "$pid_file")"
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "$label is already running with PID $old_pid"
      exit 1
    fi
  fi
done

port_pattern=":(${ros_ports[0]}|${ros_ports[1]}|${gazebo_ports[0]}|${gazebo_ports[1]})$"
if ss -ltnH | awk '{print $4}' | rg -q "$port_pattern"; then
  echo "One or more required ROS/Gazebo ports are already in use."
  exit 1
fi

mkdir -p "$LOG_DIR"
timestamp="$(date +%Y%m%d_%H%M%S)"

start_one() {
  local label="$1"
  local selection_mode="$2"
  local ros_port="$3"
  local gazebo_port="$4"
  local pid_file="$PROJECT_ROOT/.validation_dense_${label}.pid"
  local run_name="validation_dense_${label}_${timestamp}"
  local state_path="./checkpoints/${run_name}_state.pt"
  local stats_path="./results/${run_name}.npy"
  local log_file="$LOG_DIR/${run_name}.log"
  local runner_log="$LOG_DIR/${run_name}_runner.log"

  setsid bash -lc "
    set -eo pipefail
    cleanup() {
      pgid=\"\$(ps -o pgid= -p \$\$ | tr -d ' ')\"
      ps -eo pid=,pgid= | awk -v pgid=\"\$pgid\" -v self=\"\$\$\" \\
        '\$2 == pgid && \$1 != self { print \$1 }' | xargs -r kill 2>/dev/null || true
      unlink '$pid_file' 2>/dev/null || true
    }
    trap cleanup EXIT
    source /opt/ros/noetic/setup.bash
    source '$PROJECT_ROOT/env.python.sh'
    source '$PROJECT_ROOT/catkin_ws/devel_isolated/setup.bash'
    export ROS_HOSTNAME=localhost
    export ROS_MASTER_URI=http://localhost:$ros_port
    export ROS_PORT_SIM=$ros_port
    export GAZEBO_MASTER_URI=http://localhost:$gazebo_port
    export GAZEBO_RESOURCE_PATH='$PROJECT_ROOT/catkin_ws/src/multi_robot_scenario/launch'
    export DRL_MULTI_NUM_AGENTS=5
    export DRL_MULTI_SEED=$SEED
    export DRL_MULTI_TEST_LAUNCHFILE='$LAUNCHFILE'
    export DRL_MULTI_TEST_FILE_NAME='$run_name'
    export DRL_MULTI_STANDARD_ACTOR_FILE='$BASE_MODEL'
    export DRL_MULTI_ACTOR_SELECTION_MODE='$selection_mode'
    export DRL_MULTI_TEST_ACTOR_MODE=full
    export DRL_MULTI_TEST_TARGET_EPISODES=$TARGET_EPISODES
    export DRL_MULTI_TEST_STATE_PATH='$state_path'
    export DRL_MULTI_TEST_STATS_PATH='$stats_path'
    export DRL_MULTI_SCENARIO=manifest
    export DRL_MULTI_MANIFEST_PATH='$MANIFEST_PATH'
    export DRL_MULTI_MANIFEST_SAMPLING=cycle
    unset DRL_MULTI_CASE_ORACLE_MAP DRL_MULTI_RULE_ORACLE_MODE
    if [[ '$selection_mode' == 'interaction_oracle' ]]; then
      export DRL_MULTI_DENSE_ACTOR_FILE='$STRONG_MODEL'
      export DRL_MULTI_DENSE_ACTOR_MODE=full
      export DRL_MULTI_ORACLE_INTERACTION_DISTANCE=2.0
    else
      unset DRL_MULTI_DENSE_ACTOR_FILE DRL_MULTI_DENSE_ACTOR_MODE
    fi
    cd '$TD3_DIR'
    python3 -u test_velodyne_td3_multi.py >'$log_file' 2>&1
  " >"$runner_log" 2>&1 < /dev/null &

  echo $! > "$pid_file"
  echo "$label PID: $(cat "$pid_file")"
  echo "$label log: $log_file"
}

for idx in 0 1; do
  start_one \
    "${labels[$idx]}" \
    "${selection_modes[$idx]}" \
    "${ros_ports[$idx]}" \
    "${gazebo_ports[$idx]}"
done

echo "Started paired dense validation with seed $SEED."
echo "Baseline: 5A for all states."
echo "Oracle pair: 5A normally, epoch-16 Actor when another active robot is within 2 m."
echo "Both runs use the same ordered $TARGET_EPISODES-scenario dense validation manifest."
