#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
VIEW_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/robot_perception_v1"
EXPERIMENT_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/results/06_Gate开发/D5_G0_robot_detector_v1"
LOG_DIR="$PROJECT_ROOT/logs"
MODEL_NAME="TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best"
SPLIT="${1:-}"

case "$SPLIT" in
  train)
    ROS_PORT=12803
    GAZEBO_PORT=12903
    ;;
  validation)
    ROS_PORT=13003
    GAZEBO_PORT=13103
    ;;
  *)
    echo "Usage: $0 <train|validation>" >&2
    echo "The sealed test split cannot be collected through this development script." >&2
    exit 2
    ;;
esac

MANIFEST="$VIEW_DIR/$SPLIT.json.gz"
OUTPUT_DIR="$EXPERIMENT_DIR/local_data/shards/$SPLIT"
PID_FILE="$PROJECT_ROOT/.robot_perception_collection_${SPLIT}.pid"
RUN_ID="robot_perception_v1_${SPLIT}"
STATE_PATH="./checkpoints/${RUN_ID}_state.pt"
STATS_PATH="./results/${RUN_ID}.npy"

[[ -f "$MANIFEST" ]] || { echo "Perception manifest is missing: $MANIFEST"; exit 1; }
[[ -f "$TD3_DIR/pytorch_models/${MODEL_NAME}_actor.pth" ]] || {
  echo "Frozen 5A Actor is missing: $TD3_DIR/pytorch_models/${MODEL_NAME}_actor.pth"
  exit 1
}
if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Robot-perception $SPLIT collection is already running with PID $old_pid"
    exit 1
  fi
  unlink "$PID_FILE"
fi

scenario_count="$(python3 -c 'import gzip,json,sys; print(len(json.load(gzip.open(sys.argv[1], "rt"))["scenarios"]))' "$MANIFEST")"
target_episodes="${DRL_ROBOT_PERCEPTION_TARGET_EPISODES:-$scenario_count}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$LOG_DIR/collect_${RUN_ID}_${timestamp}.log"

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
  export ROS_HOSTNAME=localhost
  export ROS_MASTER_URI=http://localhost:$ROS_PORT
  export ROS_PORT_SIM=$ROS_PORT
  export GAZEBO_MASTER_URI=http://localhost:$GAZEBO_PORT
  export GAZEBO_RESOURCE_PATH='$PROJECT_ROOT/catkin_ws/src/multi_robot_scenario/launch'
  export DRL_MULTI_NUM_AGENTS=5
  export DRL_MULTI_SEED=20260727
  export DRL_MULTI_TEST_LAUNCHFILE='multi_robot_scenario_strong_interaction_pilot_5.launch'
  export DRL_MULTI_SCENARIO=manifest
  export DRL_MULTI_MANIFEST_PATH='$MANIFEST'
  export DRL_MULTI_MANIFEST_SAMPLING=cycle
  export DRL_MULTI_TEST_FILE_NAME='$MODEL_NAME'
  export DRL_MULTI_TEST_TARGET_EPISODES='$target_episodes'
  export DRL_MULTI_TEST_STATE_PATH='$STATE_PATH'
  export DRL_MULTI_TEST_STATS_PATH='$STATS_PATH'
  export DRL_MULTI_RAW_LIDAR_VOXEL_SIZE=0.01
  export DRL_MULTI_RAW_LIDAR_MAX_RANGE=6.0
  export DRL_MULTI_ROBOT_PERCEPTION_OUTPUT_DIR='$OUTPUT_DIR'
  export DRL_MULTI_ROBOT_PERCEPTION_SPLIT='$SPLIT'
  export DRL_MULTI_ROBOT_PERCEPTION_FRAME_STRIDE=2
  export DRL_MULTI_ROBOT_PERCEPTION_MAX_BACKGROUND=12
  cd '$PROJECT_ROOT/catkin_ws'
  source devel_isolated/setup.bash
  cd '$TD3_DIR'
  python3 -u test_velodyne_td3_multi.py
" >"$log_file" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "Started robot-perception $SPLIT collection."
echo "PID: $(cat "$PID_FILE")"
echo "Scenarios: $target_episodes / $scenario_count"
echo "Log: $log_file"
echo "Shards: $OUTPUT_DIR"
