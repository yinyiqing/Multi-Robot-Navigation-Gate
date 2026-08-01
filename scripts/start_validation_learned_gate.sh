#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
DATASET_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1"
EXPERIMENT_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/07_冲突拓扑组合泛化"
OUTPUT_DIR="$EXPERIMENT_DIR/local_data/G3_learned_gate_validation"
MANIFEST_PATH="$DATASET_DIR/views/dense_validation_monitor_v1/validation.json.gz"
LAUNCHFILE="multi_robot_scenario_strong_interaction_pilot_5.launch"

BASE_MODEL="TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best"
STRONG_MODEL="interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016"
DETECTOR_CHECKPOINT="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"
GATE_CHECKPOINT="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/results/06_Gate开发/D5_G2_interaction_gate_v1/local_data/model/oracle_front_v1/best.pt"

TARGET_EPISODES="${DRL_LEARNED_GATE_TARGET_EPISODES:-1}"
SEED="${DRL_LEARNED_GATE_SEED:-20260802}"
REPEAT="${DRL_LEARNED_GATE_REPEAT:-1}"
ROS_PORT="${DRL_LEARNED_GATE_ROS_PORT:-14021}"
GAZEBO_PORT="${DRL_LEARNED_GATE_GAZEBO_PORT:-14121}"
SWITCH_ON="${DRL_LEARNED_GATE_SWITCH_ON:-0.44}"
SWITCH_OFF="${DRL_LEARNED_GATE_SWITCH_OFF:-0.34}"
MINIMUM_HOLD_STEPS="${DRL_LEARNED_GATE_MINIMUM_HOLD_STEPS:-3}"
PID_FILE="$PROJECT_ROOT/.validation_learned_gate.pid"

[[ "$TARGET_EPISODES" =~ ^[1-9][0-9]*$ ]] || { echo "Target episodes must be positive"; exit 2; }
(( TARGET_EPISODES <= 200 )) || { echo "Development validation is limited to 200 scenes"; exit 2; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "Seed must be an integer"; exit 2; }
[[ "$REPEAT" =~ ^[1-9][0-9]*$ ]] || { echo "Repeat must be positive"; exit 2; }

for required in \
  "$MANIFEST_PATH" \
  "$TD3_DIR/assets/$LAUNCHFILE" \
  "$TD3_DIR/pytorch_models/${BASE_MODEL}_actor.pth" \
  "$TD3_DIR/pytorch_models/${STRONG_MODEL}_actor.pth" \
  "$DETECTOR_CHECKPOINT" \
  "$GATE_CHECKPOINT"; do
  [[ -f "$required" ]] || { echo "Required file is missing: $required"; exit 1; }
done

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Learned-Gate validation is already running with PID $old_pid"
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

mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/results" "$OUTPUT_DIR/checkpoints"
timestamp="$(date +%Y%m%d_%H%M%S)"
run_name="g3_learned_gate_n${TARGET_EPISODES}_r${REPEAT}_s${SEED}_${timestamp}"
log_file="$OUTPUT_DIR/logs/${run_name}.log"
runner_log="$OUTPUT_DIR/logs/${run_name}_runner.log"
state_path="$OUTPUT_DIR/checkpoints/${run_name}_state.pt"
stats_path="$OUTPUT_DIR/results/${run_name}.npy"

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
  export DRL_MULTI_STANDARD_ACTOR_FILE='$BASE_MODEL'
  export DRL_MULTI_DENSE_ACTOR_FILE='$STRONG_MODEL'
  export DRL_MULTI_ACTOR_SELECTION_MODE=learned_gate
  export DRL_MULTI_TEST_ACTOR_MODE=full
  export DRL_MULTI_DENSE_ACTOR_MODE=full
  export DRL_MULTI_GATE_DETECTOR_CHECKPOINT='$DETECTOR_CHECKPOINT'
  export DRL_MULTI_GATE_CHECKPOINT='$GATE_CHECKPOINT'
  export DRL_MULTI_GATE_SWITCH_ON_THRESHOLD='$SWITCH_ON'
  export DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD='$SWITCH_OFF'
  export DRL_MULTI_GATE_MINIMUM_HOLD_STEPS='$MINIMUM_HOLD_STEPS'
  export DRL_MULTI_TEST_TARGET_EPISODES=$TARGET_EPISODES
  export DRL_MULTI_TEST_STATE_PATH='$state_path'
  export DRL_MULTI_TEST_STATS_PATH='$stats_path'
  export DRL_MULTI_SCENARIO=manifest
  export DRL_MULTI_MANIFEST_PATH='$MANIFEST_PATH'
  export DRL_MULTI_MANIFEST_SAMPLING=cycle
  export DRL_MULTI_FIXED_PHYSICS_STEP_SIZE=0.001
  export DRL_MULTI_RAW_LIDAR_VOXEL_SIZE=0.01
  export DRL_MULTI_RAW_LIDAR_MAX_RANGE=6.0
  unset DRL_MULTI_CASE_ORACLE_MAP DRL_MULTI_RULE_ORACLE_MODE
  cd '$TD3_DIR'
  python3 -u test_velodyne_td3_multi.py >'$log_file' 2>&1
" >"$runner_log" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "Started learned-Gate validation."
echo "PID: $(cat "$PID_FILE")"
echo "Scenarios: $TARGET_EPISODES / 200"
echo "Gate thresholds: on=$SWITCH_ON off=$SWITCH_OFF hold=$MINIMUM_HOLD_STEPS"
echo "Log: $log_file"
echo "Stats: $stats_path"
