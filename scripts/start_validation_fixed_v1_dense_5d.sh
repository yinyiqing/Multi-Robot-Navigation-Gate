#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
DATASET_DIR="$PROJECT_ROOT/experiments/04_保留专门化/05_论文主线/datasets/fixed_v1"
LOG_DIR="$PROJECT_ROOT/logs"
PID_FILE="$PROJECT_ROOT/.validation_fixed_v1_dense_5d.pid"
ACTOR="TD3_velodyne_multi_v4_curriculum_stage2_to_5d_geo_critic_from_5a_guarded_best"
NUM_AGENTS=5
ROS_PORT=12201
GAZEBO_PORT=12301
TARGET_EPISODES=1000
SEED=20260719
MANIFEST_PATH="$DATASET_DIR/dense/validation.json.gz"
LAUNCHFILE="multi_robot_scenario_fixed_v1_dense_validation_5d_${NUM_AGENTS}.launch"

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "Dense validation manifest is missing: $MANIFEST_PATH"
  exit 1
fi
if [[ ! -f "$TD3_DIR/pytorch_models/${ACTOR}_actor.pth" ]]; then
  echo "5D actor is missing: $TD3_DIR/pytorch_models/${ACTOR}_actor.pth"
  exit 1
fi
if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Dense validation is already running with PID $old_pid"
    exit 1
  fi
fi

mkdir -p "$LOG_DIR"
python3 "$PROJECT_ROOT/scripts/generate_multi_robot_launch.py" \
  --num-agents "$NUM_AGENTS" \
  --output "$TD3_DIR/assets/$LAUNCHFILE"

timestamp="$(date +%Y%m%d_%H%M%S)"
run_name="validation1000_dense_5d_${timestamp}"
eval_log="$LOG_DIR/${run_name}.log"
runner_log="$LOG_DIR/${run_name}_runner.log"
state_path="./checkpoints/${run_name}_state.pt"
stats_path="./results/${run_name}.npy"

setsid bash -lc "
  set -eo pipefail
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
  export DRL_MULTI_SEED=$SEED
  export DRL_MULTI_TEST_LAUNCHFILE='$LAUNCHFILE'
  export DRL_MULTI_TEST_FILE_NAME='$run_name'
  export DRL_MULTI_STANDARD_ACTOR_FILE='$ACTOR'
  export DRL_MULTI_TEST_TARGET_EPISODES=$TARGET_EPISODES
  export DRL_MULTI_TEST_STATE_PATH='$state_path'
  export DRL_MULTI_TEST_STATS_PATH='$stats_path'
  export DRL_MULTI_SCENARIO=manifest
  export DRL_MULTI_MANIFEST_PATH='$MANIFEST_PATH'
  export DRL_MULTI_MANIFEST_SAMPLING=cycle
  cd '$PROJECT_ROOT/catkin_ws'
  source devel_isolated/setup.bash
  cd '$TD3_DIR'
  python3 -u test_velodyne_td3_multi.py >'$eval_log' 2>&1
" >"$runner_log" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "Started 5D on dense validation."
echo "PID: $(cat "$PID_FILE")"
echo "Episodes: $TARGET_EPISODES"
echo "Evaluation log: $eval_log"
echo "Runner log: $runner_log"
