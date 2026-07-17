#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
LOG_DIR="$PROJECT_ROOT/logs"
DATASET_DIR="$PROJECT_ROOT/experiments/04_保留专门化/05_论文主线/datasets/fixed_v1"
MODEL_NAME="TD3_velodyne_multi_v4_curriculum_stage2_to_5d_geo_critic_from_5a_guarded_best"
NUM_AGENTS=5

POOL="${1:-}"
case "$POOL" in
  standard)
    TARGET_EPISODES=1000
    DEFAULT_ROS_PORT=11601
    DEFAULT_GAZEBO_PORT=11701
    SEED=20260717
    ;;
  dense)
    TARGET_EPISODES=2000
    DEFAULT_ROS_PORT=11602
    DEFAULT_GAZEBO_PORT=11702
    SEED=20260718
    ;;
  *)
    echo "Usage: $0 standard|dense"
    exit 2
    ;;
esac

MANIFEST_PATH="$DATASET_DIR/$POOL/test.json.gz"
PID_FILE="$PROJECT_ROOT/.test_fixed_v1_${POOL}_5d.pid"
LAUNCHFILE="multi_robot_scenario_dense_fixed_v1_${POOL}_${NUM_AGENTS}.launch"
LAUNCH_PATH="$TD3_DIR/assets/$LAUNCHFILE"
ROS_PORT="${DRL_MULTI_TEST_ROS_PORT:-$DEFAULT_ROS_PORT}"
GAZEBO_PORT="${DRL_MULTI_TEST_GAZEBO_PORT:-$DEFAULT_GAZEBO_PORT}"
STATE_PATH="./checkpoints/${MODEL_NAME}_fixed_v1_${POOL}_test_state.pt"
STATS_PATH="./results/${MODEL_NAME}_fixed_v1_${POOL}_test.npy"

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "Manifest is missing: $MANIFEST_PATH"
  exit 1
fi
if [[ ! -f "$TD3_DIR/pytorch_models/${MODEL_NAME}_actor.pth" ]]; then
  echo "Actor is missing: $TD3_DIR/pytorch_models/${MODEL_NAME}_actor.pth"
  exit 1
fi
if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "$POOL fixed-v1 test is already running with PID $old_pid"
    exit 1
  fi
fi

mkdir -p "$LOG_DIR"
python3 "$PROJECT_ROOT/scripts/generate_multi_robot_launch.py" \
  --num-agents "$NUM_AGENTS" \
  --output "$LAUNCH_PATH"

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$LOG_DIR/test_fixed_v1_${POOL}_5d_${timestamp}.log"

setsid bash -lc "
  source /opt/ros/noetic/setup.bash
  source '$PROJECT_ROOT/env.python.sh'
  export ROS_HOSTNAME=localhost
  export ROS_MASTER_URI=http://localhost:${ROS_PORT}
  export ROS_PORT_SIM=${ROS_PORT}
  export GAZEBO_MASTER_URI=http://localhost:${GAZEBO_PORT}
  export GAZEBO_RESOURCE_PATH='$PROJECT_ROOT/catkin_ws/src/multi_robot_scenario/launch'
  export DRL_MULTI_NUM_AGENTS=${NUM_AGENTS}
  export DRL_MULTI_SEED=${SEED}
  export DRL_MULTI_TEST_LAUNCHFILE='$LAUNCHFILE'
  export DRL_MULTI_TEST_FILE_NAME='$MODEL_NAME'
  export DRL_MULTI_TEST_TARGET_EPISODES=${TARGET_EPISODES}
  export DRL_MULTI_TEST_STATE_PATH='$STATE_PATH'
  export DRL_MULTI_TEST_STATS_PATH='$STATS_PATH'
  export DRL_MULTI_SCENARIO=manifest
  export DRL_MULTI_MANIFEST_PATH='$MANIFEST_PATH'
  export DRL_MULTI_MANIFEST_SAMPLING=cycle
  cd '$PROJECT_ROOT/catkin_ws'
  source devel_isolated/setup.bash
  cd '$TD3_DIR'
  exec python3 -u test_velodyne_td3_multi.py
" >"$log_file" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "Started fixed-v1 $POOL 5D test."
echo "PID: $(cat "$PID_FILE")"
echo "Episodes: $TARGET_EPISODES"
echo "Manifest: $MANIFEST_PATH"
echo "ROS/Gazebo ports: $ROS_PORT/$GAZEBO_PORT"
echo "Log: $log_file"
