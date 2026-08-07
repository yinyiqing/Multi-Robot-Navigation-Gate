#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
MODEL_NAME="capacity_wide_r2_s1_repair_n1_seed20260813"
MANIFEST="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/g12_r2_curriculum_v1/n1/validation.json.gz"
RUN_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/12_参数匹配单Actor容量对照/local_data/s1_repair_validation"
LOG_DIR="$PROJECT_ROOT/logs/active/capacity-wide-g12-r2/s1-repair"
LAUNCHFILE="$LOG_DIR/runtime_g12_r2_s1_repair_validation.launch"
PID_FILE="$PROJECT_ROOT/.g12_r2_s1_repair_validation.pid"
LOCK_FILE="/tmp/local_critic_multi_robot_training.lock"
VALIDATION_LOG="${1:?validation log path is required}"
ROS_PORT=14641
GAZEBO_PORT=14741

stop_runtime_children() {
  local pgid child_pids
  pgid="$(ps -o pgid= -p $$ | tr -d ' ')"
  child_pids="$(
    ps -eo pid=,pgid= | awk -v pgid="$pgid" -v self="$$" \
      '$2 == pgid && $1 != self { print $1 }'
  )"
  if [[ -n "$child_pids" ]]; then
    xargs -r kill -TERM 2>/dev/null <<<"$child_pids" || true
    sleep 3
    child_pids="$(
      ps -eo pid=,pgid= | awk -v pgid="$pgid" -v self="$$" \
        '$2 == pgid && $1 != self { print $1 }'
    )"
    [[ -z "$child_pids" ]] || xargs -r kill -KILL 2>/dev/null <<<"$child_pids" || true
  fi
  fuser -k -TERM "${ROS_PORT}/tcp" "${GAZEBO_PORT}/tcp" >/dev/null 2>&1 || true
}

cleanup() {
  stop_runtime_children
  unlink "$PID_FILE" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

exec 9>"$LOCK_FILE"
flock -n 9 || { echo "Multi-robot validation lock is busy" >&2; exit 1; }
mkdir -p "$RUN_DIR" "$LOG_DIR"
/usr/bin/python3 "$PROJECT_ROOT/scripts/generate_multi_robot_launch.py" \
  --num-agents 1 --output "$LAUNCHFILE"

set +u
source /opt/ros/noetic/setup.bash
source "$PROJECT_ROOT/env.python.sh"
source "$PROJECT_ROOT/catkin_ws/devel_isolated/setup.bash"
set -u

export CUDA_VISIBLE_DEVICES=0
export ROS_HOSTNAME=localhost
export GAZEBO_IP=127.0.0.1
export ROS_MASTER_URI="http://localhost:$ROS_PORT"
export ROS_PORT_SIM="$ROS_PORT"
export GAZEBO_MASTER_URI="http://localhost:$GAZEBO_PORT"
export GAZEBO_RESOURCE_PATH="$PROJECT_ROOT/catkin_ws/src/multi_robot_scenario/launch"
export DRL_MULTI_NUM_AGENTS=1
export DRL_MULTI_SEED=20260813
export DRL_MULTI_TEST_LAUNCHFILE="$LAUNCHFILE"
export DRL_MULTI_TEST_FILE_NAME="$MODEL_NAME"
export DRL_MULTI_TEST_ACTOR_MODE=full
export DRL_MULTI_ACTOR_SELECTION_MODE=single
export DRL_MULTI_TEST_TARGET_EPISODES=120
export DRL_MULTI_TEST_STATE_PATH="$RUN_DIR/state.pt"
export DRL_MULTI_TEST_STATS_PATH="$RUN_DIR/results.npy"
export DRL_MULTI_SCENARIO=manifest
export DRL_MULTI_MANIFEST_PATH="$MANIFEST"
export DRL_MULTI_MANIFEST_SAMPLING=cycle
export DRL_MULTI_FIXED_PHYSICS_STEP_SIZE=0.001
export DRL_MULTI_REQUIRE_FIXED_STEP_SERVICE=1

echo "Starting G12-R2-S1 repair broad validation"
echo "Actor: $TD3_DIR/pytorch_models/${MODEL_NAME}_actor.pth"
echo "Manifest: $MANIFEST"
echo "Validation log: $VALIDATION_LOG"
cd "$TD3_DIR"
python3 -u test_velodyne_td3_multi.py >>"$VALIDATION_LOG" 2>&1
/usr/bin/python3 "$PROJECT_ROOT/scripts/analyze_g12_r2_s1_repair_validation.py" \
  --stats "$RUN_DIR/results.npy" \
  --manifest "$MANIFEST" \
  --output "$RUN_DIR/summary.json" >>"$VALIDATION_LOG" 2>&1
echo "G12-R2-S1 repair broad validation complete."
