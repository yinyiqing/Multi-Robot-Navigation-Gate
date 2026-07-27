#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
VIEW_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/strong_interaction_curriculum_v1"
LOG_DIR="$PROJECT_ROOT/logs"
MANIFEST_PATH="$VIEW_DIR/validation.json.gz"
LAUNCHFILE="multi_robot_scenario_strong_interaction_pilot_5.launch"
BASE_MODEL="${DRL_MULTI_BASE_MODEL:-TD3_velodyne_multi_v4_curriculum_stage2_to_5d_geo_critic_from_5a_guarded_best}"
STRONG_MODEL="${DRL_MULTI_STRONG_MODEL:-interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_best}"
REPEAT="${1:-1}"

[[ "$REPEAT" =~ ^[1-9][0-9]*$ ]] || { echo "Repeat must be a positive integer."; exit 2; }
[[ -f "$MANIFEST_PATH" ]] || { echo "Validation manifest is missing: $MANIFEST_PATH"; exit 1; }
[[ -f "$TD3_DIR/assets/$LAUNCHFILE" ]] || { echo "Launch file is missing: $LAUNCHFILE"; exit 1; }
for model in "$BASE_MODEL" "$STRONG_MODEL"; do
  [[ -f "$TD3_DIR/pytorch_models/${model}_actor.pth" ]] || {
    echo "Actor is missing: $TD3_DIR/pytorch_models/${model}_actor.pth"
    exit 1
  }
done

seed=$((20260727 + REPEAT))
base_ros_port=$((13400 + REPEAT * 10))
base_gazebo_port=$((13500 + REPEAT * 10))
labels=("5d" "strong_e16")
ros_ports=("$base_ros_port" "$((base_ros_port + 1))")
gazebo_ports=("$base_gazebo_port" "$((base_gazebo_port + 1))")
models=("$BASE_MODEL" "$STRONG_MODEL")

for idx in 0 1; do
  label="${labels[$idx]}"
  pid_file="$PROJECT_ROOT/.validation_strong_actor_pair_${label}_r${REPEAT}.pid"
  state_file="$TD3_DIR/checkpoints/standalone_pair_${label}_r${REPEAT}_state.pt"
  stats_file="$TD3_DIR/results/standalone_pair_${label}_r${REPEAT}.npy"
  if [[ -f "$pid_file" ]]; then
    old_pid="$(tr -d '[:space:]' < "$pid_file")"
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "$label repeat $REPEAT is already running with PID $old_pid"
      exit 1
    fi
  fi
  [[ ! -e "$state_file" && ! -e "$stats_file" ]] || {
    echo "$label repeat $REPEAT already has output; use a new repeat number."
    exit 1
  }
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
  local model="$2"
  local ros_port="$3"
  local gazebo_port="$4"
  local pid_file="$PROJECT_ROOT/.validation_strong_actor_pair_${label}_r${REPEAT}.pid"
  local state_path="./checkpoints/standalone_pair_${label}_r${REPEAT}_state.pt"
  local stats_path="./results/standalone_pair_${label}_r${REPEAT}.npy"
  local log_file="$LOG_DIR/validation_standalone_pair_${label}_r${REPEAT}_${timestamp}.log"

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
    export DRL_MULTI_SEED=$seed
    export DRL_MULTI_TEST_LAUNCHFILE='$LAUNCHFILE'
    export DRL_MULTI_TEST_FILE_NAME='standalone_pair_${label}_r${REPEAT}'
    export DRL_MULTI_STANDARD_ACTOR_FILE='$model'
    export DRL_MULTI_ACTOR_SELECTION_MODE=single
    export DRL_MULTI_TEST_ACTOR_MODE=full
    export DRL_MULTI_TEST_TARGET_EPISODES=140
    export DRL_MULTI_TEST_STATE_PATH='$state_path'
    export DRL_MULTI_TEST_STATS_PATH='$stats_path'
    export DRL_MULTI_SCENARIO=manifest
    export DRL_MULTI_MANIFEST_PATH='$MANIFEST_PATH'
    export DRL_MULTI_MANIFEST_SAMPLING=cycle
    unset DRL_MULTI_DENSE_ACTOR_FILE DRL_MULTI_RULE_ORACLE_MODE
    cd '$TD3_DIR'
    python3 -u test_velodyne_td3_multi.py
  " >"$log_file" 2>&1 < /dev/null &

  echo $! > "$pid_file"
  echo "$label PID: $(cat "$pid_file")"
  echo "$label log: $log_file"
}

for idx in 0 1; do
  start_one \
    "${labels[$idx]}" \
    "${models[$idx]}" \
    "${ros_ports[$idx]}" \
    "${gazebo_ports[$idx]}"
done

echo "Started standalone paired validation repeat $REPEAT with seed $seed."
echo "5D: $BASE_MODEL"
echo "Strong Actor: $STRONG_MODEL"
echo "Both use the same ordered 140-scenario validation manifest."
echo "Expected runtime: roughly 20-35 minutes."
