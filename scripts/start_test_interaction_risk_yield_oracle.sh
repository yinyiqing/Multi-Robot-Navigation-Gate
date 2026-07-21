#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
LOG_DIR="$PROJECT_ROOT/logs"
VIEW_DIR="$PROJECT_ROOT/experiments/04_保留专门化/05_论文主线/datasets/fixed_v1/views/interaction_risk_v1"
MODEL_NAME="TD3_velodyne_multi_v4_curriculum_stage2_to_5d_geo_critic_from_5a_guarded_best"
RUN_NAME="interaction_risk_yield_oracle_v1_s20260721"
NUM_AGENTS=5
TARGET_EPISODES=60
ROS_PORT="${DRL_MULTI_TEST_ROS_PORT:-12622}"
GAZEBO_PORT="${DRL_MULTI_TEST_GAZEBO_PORT:-12722}"
PID_FILE="$PROJECT_ROOT/.test_interaction_risk_yield_oracle.pid"
LAUNCHFILE="multi_robot_scenario_interaction_risk_yield_oracle_${NUM_AGENTS}.launch"
MANIFEST_PATH="$VIEW_DIR/probe.json.gz"
STATE_PATH="./checkpoints/${RUN_NAME}_state.pt"
STATS_PATH="./results/${RUN_NAME}.npy"
TRAJECTORY_PATH="$LOG_DIR/${RUN_NAME}_trajectory.jsonl"

for required in \
  "$MANIFEST_PATH" \
  "$TD3_DIR/pytorch_models/${MODEL_NAME}_actor.pth"; do
  if [[ ! -f "$required" ]]; then
    echo "Required file is missing: $required"
    exit 1
  fi
done

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Interaction-risk yield oracle is already running with PID $old_pid"
    exit 1
  fi
fi

for output in "$TD3_DIR/${STATE_PATH#./}" "$TD3_DIR/${STATS_PATH#./}" "$TRAJECTORY_PATH"; do
  if [[ -e "$output" ]]; then
    echo "Refusing to resume or overwrite existing oracle output: $output"
    exit 1
  fi
done

mkdir -p "$LOG_DIR"
python3 "$PROJECT_ROOT/scripts/generate_multi_robot_launch.py" \
  --num-agents "$NUM_AGENTS" \
  --output "$TD3_DIR/assets/$LAUNCHFILE"

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$LOG_DIR/test_${RUN_NAME}_${timestamp}.log"

setsid bash -lc "
  cleanup() {
    pgid=\"\$(ps -o pgid= -p \$\$ | tr -d ' ')\"
    ps -eo pid=,pgid= | awk -v pgid=\"\$pgid\" -v self=\"\$\$\" \\
      '\$2 == pgid && \$1 != self { print \$1 }' | xargs -r kill 2>/dev/null || true
    rm -f '$PID_FILE'
  }
  trap cleanup EXIT
  source /opt/ros/noetic/setup.bash
  source '$PROJECT_ROOT/env.python.sh'
  export ROS_HOSTNAME=localhost
  export ROS_MASTER_URI=http://localhost:$ROS_PORT
  export ROS_PORT_SIM=$ROS_PORT
  export GAZEBO_MASTER_URI=http://localhost:$GAZEBO_PORT
  export GAZEBO_RESOURCE_PATH='$PROJECT_ROOT/catkin_ws/src/multi_robot_scenario/launch'
  export DRL_MULTI_NUM_AGENTS=$NUM_AGENTS
  export DRL_MULTI_SEED=20260721
  export DRL_MULTI_TEST_LAUNCHFILE='$LAUNCHFILE'
  export DRL_MULTI_TEST_FILE_NAME='$MODEL_NAME'
  export DRL_MULTI_TEST_TARGET_EPISODES=$TARGET_EPISODES
  export DRL_MULTI_TEST_STATE_PATH='$STATE_PATH'
  export DRL_MULTI_TEST_STATS_PATH='$STATS_PATH'
  export DRL_MULTI_TRAJECTORY_JSONL='$TRAJECTORY_PATH'
  export DRL_MULTI_RULE_ORACLE_MODE=conflict_pair_yield
  export DRL_MULTI_RULE_ORACLE_STOP_DISTANCE=1.2
  export DRL_MULTI_RULE_ORACLE_RELEASE_DISTANCE=1.4
  export DRL_MULTI_RULE_ORACLE_MAX_YIELD_STEPS=20
  export DRL_MULTI_SCENARIO=manifest
  export DRL_MULTI_MANIFEST_PATH='$MANIFEST_PATH'
  export DRL_MULTI_MANIFEST_SAMPLING=cycle
  cd '$PROJECT_ROOT/catkin_ws'
  source devel_isolated/setup.bash
  cd '$TD3_DIR'
  python3 -u test_velodyne_td3_multi.py
" >"$log_file" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "Started interaction-risk fixed-priority yield oracle."
echo "PID: $(cat "$PID_FILE")"
echo "Episodes: $TARGET_EPISODES"
echo "Estimated time: 8-15 minutes"
echo "Log: $log_file"
echo "Trajectory: $TRAJECTORY_PATH"
