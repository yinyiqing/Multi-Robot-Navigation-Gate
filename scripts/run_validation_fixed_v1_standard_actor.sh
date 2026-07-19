#!/usr/bin/env bash
set -eo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 LABEL ACTOR_PREFIX ROS_PORT GAZEBO_PORT LOG_FILE"
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
DATASET_DIR="$PROJECT_ROOT/experiments/04_保留专门化/05_论文主线/datasets/fixed_v1"
LABEL="$1"
ACTOR_PREFIX="$2"
ROS_PORT="$3"
GAZEBO_PORT="$4"
LOG_FILE="$5"
NUM_AGENTS=5
TARGET_EPISODES=500
SEED=20260719
MANIFEST_PATH="$DATASET_DIR/standard/validation.json.gz"
LAUNCHFILE="multi_robot_scenario_fixed_v1_validation_${LABEL}_${NUM_AGENTS}.launch"
RUN_NAME="validation500_standard_${LABEL}"
STATE_PATH="./checkpoints/${RUN_NAME}_state.pt"
STATS_PATH="./results/${RUN_NAME}.npy"

if [[ ! "$LABEL" =~ ^[a-z0-9_]+$ ]]; then
  echo "LABEL must contain only lowercase letters, digits, and underscores"
  exit 2
fi
if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "Validation manifest is missing: $MANIFEST_PATH"
  exit 1
fi
if [[ ! -f "$TD3_DIR/pytorch_models/${ACTOR_PREFIX}_actor.pth" ]]; then
  echo "Actor is missing: $TD3_DIR/pytorch_models/${ACTOR_PREFIX}_actor.pth"
  exit 1
fi

python3 "$PROJECT_ROOT/scripts/generate_multi_robot_launch.py" \
  --num-agents "$NUM_AGENTS" \
  --output "$TD3_DIR/assets/$LAUNCHFILE"

source /opt/ros/noetic/setup.bash
source "$PROJECT_ROOT/env.python.sh"
export ROS_HOSTNAME=localhost
export ROS_MASTER_URI="http://localhost:${ROS_PORT}"
export ROS_PORT_SIM="$ROS_PORT"
export GAZEBO_MASTER_URI="http://localhost:${GAZEBO_PORT}"
export GAZEBO_RESOURCE_PATH="$PROJECT_ROOT/catkin_ws/src/multi_robot_scenario/launch"
export DRL_MULTI_NUM_AGENTS="$NUM_AGENTS"
export DRL_MULTI_SEED="$SEED"
export DRL_MULTI_TEST_LAUNCHFILE="$LAUNCHFILE"
export DRL_MULTI_TEST_FILE_NAME="$RUN_NAME"
export DRL_MULTI_STANDARD_ACTOR_FILE="$ACTOR_PREFIX"
export DRL_MULTI_TEST_TARGET_EPISODES="$TARGET_EPISODES"
export DRL_MULTI_TEST_STATE_PATH="$STATE_PATH"
export DRL_MULTI_TEST_STATS_PATH="$STATS_PATH"
export DRL_MULTI_SCENARIO=manifest
export DRL_MULTI_MANIFEST_PATH="$MANIFEST_PATH"
export DRL_MULTI_MANIFEST_SAMPLING=cycle

cd "$PROJECT_ROOT/catkin_ws"
source devel_isolated/setup.bash
cd "$TD3_DIR"
exec python3 -u test_velodyne_td3_multi.py >"$LOG_FILE" 2>&1
