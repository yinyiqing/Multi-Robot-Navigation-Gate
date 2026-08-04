#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
VIEW_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/g11_a1_gate_v1"
RUN_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究/G11_A1_当前协议时序pilot"
LOG_DIR="$PROJECT_ROOT/logs/active/g11_a1"
MODEL_NAME="TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best"
EXPECTED_ACTOR_SHA="fa28855049b67b3ee44c66d55d4f14441fc7c521e5429862c75b152f7d5cacc5"
PROFILE="${1:-}"

case "$PROFILE" in
  train)
    MANIFEST="$VIEW_DIR/train.json.gz"
    EXPECTED_MANIFEST_SHA="a97534ed22d1b4b12951cd42b80d515037774830f38cb432926c0e9f50379026"
    SPLIT=train
    ROS_PORT=13623
    GAZEBO_PORT=13723
    ;;
  validation)
    MANIFEST="$VIEW_DIR/validation.json.gz"
    EXPECTED_MANIFEST_SHA="e261a7afbac8f7341ab13609c2662a2824a0ff383789287ad7733290389cd99d"
    SPLIT=validation
    ROS_PORT=13823
    GAZEBO_PORT=13923
    ;;
  *)
    echo "Usage: $0 <train|validation>" >&2
    echo "No navigation validation/test or sealed-test profile is exposed." >&2
    exit 2
    ;;
esac

ACTOR="$TD3_DIR/pytorch_models/${MODEL_NAME}_actor.pth"
OUTPUT_DIR="$RUN_DIR/local_data/shards/$SPLIT"
PID_FILE="$PROJECT_ROOT/.robot_perception_g11_a1_${PROFILE}.pid"
RUN_ID="g11_a1_gate_${PROFILE}"
STATE_PATH="./checkpoints/${RUN_ID}_state.pt"
STATS_PATH="./results/${RUN_ID}.npy"

for required in "$MANIFEST" "$ACTOR"; do
  [[ -f "$required" ]] || { echo "Required input is missing: $required" >&2; exit 1; }
done
[[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" == "$EXPECTED_MANIFEST_SHA" ]] || {
  echo "G11-A1 $PROFILE manifest hash mismatch" >&2
  exit 1
}
[[ "$(sha256sum "$ACTOR" | awk '{print $1}')" == "$EXPECTED_ACTOR_SHA" ]] || {
  echo "Frozen 5A Actor hash mismatch" >&2
  exit 1
}
if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "G11-A1 $PROFILE collection is already running with PID $old_pid" >&2
    exit 1
  fi
  unlink "$PID_FILE"
fi

scenario_count="$(/usr/bin/python3 -c 'import gzip,json,sys; print(len(json.load(gzip.open(sys.argv[1], "rt"))["scenarios"]))' "$MANIFEST")"
target_episodes="${DRL_G11_A1_TARGET_EPISODES:-$scenario_count}"
if ! [[ "$target_episodes" =~ ^[1-9][0-9]*$ ]] || (( target_episodes > scenario_count )); then
  echo "DRL_G11_A1_TARGET_EPISODES must be in [1, $scenario_count]" >&2
  exit 2
fi
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
  source '$PROJECT_ROOT/catkin_ws/devel_isolated/setup.bash'
  export CUDA_VISIBLE_DEVICES=''
  export ROS_HOSTNAME=localhost
  export ROS_MASTER_URI=http://localhost:$ROS_PORT
  export ROS_PORT_SIM=$ROS_PORT
  export GAZEBO_MASTER_URI=http://localhost:$GAZEBO_PORT
  export GAZEBO_RESOURCE_PATH='$PROJECT_ROOT/catkin_ws/src/multi_robot_scenario/launch'
  export DRL_MULTI_NUM_AGENTS=5
  export DRL_MULTI_SEED=20260804
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
  cd '$TD3_DIR'
  nice -n 10 python3 -u test_velodyne_td3_multi.py
" >"$log_file" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "Started G11-A1 $PROFILE collection."
echo "PID: $(cat "$PID_FILE")"
echo "Scenarios: $target_episodes / $scenario_count"
echo "Log: $log_file"
echo "Shards: $OUTPUT_DIR"
