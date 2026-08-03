#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
VIEW_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/strong_interaction_curriculum_v1"
LOG_DIR="$PROJECT_ROOT/logs"
MANIFEST_PATH="$VIEW_DIR/validation.json.gz"
LAUNCHFILE="multi_robot_scenario_strong_interaction_pilot_5.launch"
BASE_MODEL="${DRL_MULTI_BASE_MODEL:-TD3_velodyne_multi_v4_curriculum_stage2_to_5d_geo_critic_from_5a_guarded_best}"
CANDIDATE_MODEL="${DRL_MULTI_CANDIDATE_MODEL:-interaction_oracle_specialist_pilot_s20260724_epoch_002}"
BASE_LABEL="${DRL_MULTI_BASE_LABEL:-5d}"
CANDIDATE_LABEL="${DRL_MULTI_CANDIDATE_LABEL:-epoch2}"
REPEAT="${1:-1}"

[[ "$REPEAT" =~ ^[1-9][0-9]*$ ]] || { echo "Repeat must be a positive integer."; exit 2; }
[[ "$BASE_LABEL" =~ ^[A-Za-z0-9_]+$ ]] || { echo "Base label contains invalid characters."; exit 2; }
[[ "$CANDIDATE_LABEL" =~ ^[A-Za-z0-9_]+$ ]] || { echo "Candidate label contains invalid characters."; exit 2; }
[[ -f "$MANIFEST_PATH" ]] || { echo "Validation manifest is missing: $MANIFEST_PATH"; exit 1; }
[[ -f "$TD3_DIR/assets/$LAUNCHFILE" ]] || { echo "Launch file is missing: $LAUNCHFILE"; exit 1; }
for model in "$BASE_MODEL" "$CANDIDATE_MODEL"; do
  [[ -f "$TD3_DIR/pytorch_models/${model}_actor.pth" ]] || {
    echo "Actor is missing: $TD3_DIR/pytorch_models/${model}_actor.pth"
    exit 1
  }
done

seed=$((20260724 + REPEAT))
base_ros_port=$((13200 + REPEAT * 10))
base_gazebo_port=$((13300 + REPEAT * 10))
mkdir -p "$LOG_DIR"
timestamp="$(date +%Y%m%d_%H%M%S)"

start_one() {
  local label="$1"
  local ros_port="$2"
  local gazebo_port="$3"
  local selection_mode="$4"
  local dense_model="$5"
  local pid_file="$PROJECT_ROOT/.test_interaction_oracle_${label}_r${REPEAT}.pid"
  local state_path="./checkpoints/interaction_oracle_eval_${label}_r${REPEAT}_state.pt"
  local stats_path="./results/interaction_oracle_eval_${label}_r${REPEAT}.npy"
  local log_file="$LOG_DIR/test_interaction_oracle_${label}_r${REPEAT}_${timestamp}.log"

  if [[ -f "$pid_file" ]]; then
    local old_pid
    old_pid="$(tr -d '[:space:]' < "$pid_file")"
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "$label repeat $REPEAT is already running with PID $old_pid"
      exit 1
    fi
  fi
  [[ ! -e "$TD3_DIR/$state_path" && ! -e "$TD3_DIR/$stats_path" ]] || {
    echo "$label repeat $REPEAT already has output; use a new repeat number."
    exit 1
  }
  if ss -ltn | awk '{print $4}' | grep -Eq ":${ros_port}$|:${gazebo_port}$"; then
    echo "Port $ros_port or $gazebo_port is already in use."
    exit 1
  fi

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
    export DRL_MULTI_TEST_FILE_NAME='interaction_oracle_eval_${label}_r${REPEAT}'
    export DRL_MULTI_STANDARD_ACTOR_FILE='$BASE_MODEL'
    export DRL_MULTI_DENSE_ACTOR_FILE='$dense_model'
    export DRL_MULTI_ACTOR_SELECTION_MODE='$selection_mode'
    export DRL_MULTI_ORACLE_INTERACTION_DISTANCE=2.0
    export DRL_MULTI_TEST_TARGET_EPISODES=140
    export DRL_MULTI_TEST_STATE_PATH='$state_path'
    export DRL_MULTI_TEST_STATS_PATH='$stats_path'
    export DRL_MULTI_SCENARIO=manifest
    export DRL_MULTI_MANIFEST_PATH='$MANIFEST_PATH'
    export DRL_MULTI_MANIFEST_SAMPLING=cycle
    cd '$TD3_DIR'
    python3 -u test_velodyne_td3_multi.py
  " >"$log_file" 2>&1 < /dev/null &

  echo $! > "$pid_file"
  echo "$label PID: $(cat "$pid_file")"
  echo "$label log: $log_file"
}

start_one "$BASE_LABEL" "$base_ros_port" "$base_gazebo_port" "single" ""
start_one "$CANDIDATE_LABEL" "$((base_ros_port + 1))" "$((base_gazebo_port + 1))" \
  "interaction_oracle" "$CANDIDATE_MODEL"

echo "Started paired repeat $REPEAT with seed $seed."
echo "Baseline: $BASE_MODEL"
echo "Candidate: $CANDIDATE_MODEL"
echo "Both tests use the same ordered 140-scenario validation manifest."
echo "Expected runtime: roughly 15-25 minutes."
