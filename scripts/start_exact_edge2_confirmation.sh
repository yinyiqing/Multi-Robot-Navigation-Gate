#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
EXPERIMENT_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/08_exact_edge2零样本确认"
OUTPUT_DIR="$EXPERIMENT_DIR/local_data"
MANIFEST_PATH="$EXPERIMENT_DIR/validation.json"
LAUNCHFILE="multi_robot_scenario_strong_interaction_pilot_5.launch"
BASE_MODEL="TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best"
STRONG_MODEL="interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016"
DETECTOR_CHECKPOINT="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"
GATE_CHECKPOINT="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/results/06_Gate开发/D5_G2_interaction_gate_v1/local_data/model/oracle_front_v1/best.pt"
MODE="${1:-}"

case "$MODE" in
  5a)
    ROS_PORT=15021
    GAZEBO_PORT=15121
    ;;
  learned-gate)
    ROS_PORT=15221
    GAZEBO_PORT=15321
    ;;
  *)
    echo "Usage: $0 <5a|learned-gate>" >&2
    exit 2
    ;;
esac

TARGET_EPISODES="${DRL_EDGE2_TARGET_EPISODES:-1}"
SEED="${DRL_EDGE2_SEED:-20260803}"
RESUME_RUN_NAME="${DRL_EDGE2_RUN_NAME:-}"
PID_FILE="$PROJECT_ROOT/.edge2_confirmation_${MODE//-/_}.pid"
[[ "$TARGET_EPISODES" =~ ^[1-9][0-9]*$ ]] || { echo "Target episodes must be positive"; exit 2; }
(( TARGET_EPISODES <= 200 )) || { echo "Edge-2 confirmation is limited to 200 scenes"; exit 2; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "Seed must be an integer"; exit 2; }

required_files=(
  "$MANIFEST_PATH"
  "$TD3_DIR/assets/$LAUNCHFILE"
  "$TD3_DIR/pytorch_models/${BASE_MODEL}_actor.pth"
)
if [[ "$MODE" == "learned-gate" ]]; then
  required_files+=(
    "$TD3_DIR/pytorch_models/${STRONG_MODEL}_actor.pth"
    "$DETECTOR_CHECKPOINT"
    "$GATE_CHECKPOINT"
  )
fi
for required in "${required_files[@]}"; do
  [[ -f "$required" ]] || { echo "Required file is missing: $required"; exit 1; }
done

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Edge-2 $MODE confirmation is already running with PID $old_pid"
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

mkdir -p "$OUTPUT_DIR/$MODE/logs" "$OUTPUT_DIR/$MODE/results" "$OUTPUT_DIR/$MODE/checkpoints"
timestamp="$(date +%Y%m%d_%H%M%S)"
run_name="${RESUME_RUN_NAME:-edge2_${MODE//-/_}_n${TARGET_EPISODES}_s${SEED}_${timestamp}}"
log_suffix=""
if [[ -n "$RESUME_RUN_NAME" ]]; then
  log_suffix="_resume_${timestamp}"
fi
log_file="$OUTPUT_DIR/$MODE/logs/${run_name}${log_suffix}.log"
runner_log="$OUTPUT_DIR/$MODE/logs/${run_name}${log_suffix}_runner.log"
state_path="$OUTPUT_DIR/$MODE/checkpoints/${run_name}_state.pt"
stats_path="$OUTPUT_DIR/$MODE/results/${run_name}.npy"
if [[ -n "$RESUME_RUN_NAME" ]]; then
  [[ -f "$state_path" ]] || { echo "Resume state is missing: $state_path"; exit 1; }
  [[ -f "$stats_path" ]] || { echo "Resume results are missing: $stats_path"; exit 1; }
fi

dense_exports="
  export DRL_MULTI_DENSE_ACTOR_FILE=''
  export DRL_MULTI_ACTOR_SELECTION_MODE=single
"
if [[ "$MODE" == "learned-gate" ]]; then
  dense_exports="
  export DRL_MULTI_DENSE_ACTOR_FILE='$STRONG_MODEL'
  export DRL_MULTI_ACTOR_SELECTION_MODE=learned_gate
  export DRL_MULTI_DENSE_ACTOR_MODE=full
  export DRL_MULTI_GATE_DETECTOR_CHECKPOINT='$DETECTOR_CHECKPOINT'
  export DRL_MULTI_GATE_CHECKPOINT='$GATE_CHECKPOINT'
  export DRL_MULTI_GATE_SWITCH_ON_THRESHOLD=0.44
  export DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD=0.34
  export DRL_MULTI_GATE_MINIMUM_HOLD_STEPS=3
  "
fi

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
  export CUDA_VISIBLE_DEVICES=''
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
  export DRL_MULTI_TEST_ACTOR_MODE=full
  $dense_exports
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
echo "Started exact-edge-2 $MODE confirmation."
echo "PID: $(cat "$PID_FILE")"
echo "Scenarios: $TARGET_EPISODES / 200"
echo "Seed/device: $SEED/cpu"
echo "Resume run: ${RESUME_RUN_NAME:-disabled}"
echo "Log: $log_file"
echo "Stats: $stats_path"
