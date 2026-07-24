#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
VIEW_DIR="$PROJECT_ROOT/experiments/04_保留专门化/05_论文主线/datasets/fixed_v1/views/interaction_risk_v1"
LOG_DIR="$PROJECT_ROOT/logs"
PID_FILE="$PROJECT_ROOT/.test_lidar_cluster_sensor_probe_5d.pid"
MODEL_NAME="TD3_velodyne_multi_v4_curriculum_stage2_to_5d_geo_critic_from_5a_guarded_best"
RUN_ID="lidar_cluster_shape_probe_5d_s20260724"
MANIFEST="$VIEW_DIR/sensor_probe.json.gz"
ROS_PORT=12603
GAZEBO_PORT=12703

[[ -f "$MANIFEST" ]] || { echo "Sensor probe manifest is missing: $MANIFEST"; exit 1; }
[[ -f "$TD3_DIR/pytorch_models/${MODEL_NAME}_actor.pth" ]] || {
  echo "5D Actor is missing: $TD3_DIR/pytorch_models/${MODEL_NAME}_actor.pth"
  exit 1
}
if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Lidar cluster sensor probe is already running with PID $old_pid"
    exit 1
  fi
  unlink "$PID_FILE"
fi

mkdir -p "$LOG_DIR"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$LOG_DIR/test_${RUN_ID}_${timestamp}.log"
trajectory_file="$LOG_DIR/${RUN_ID}_${timestamp}.jsonl"
run_tag="${RUN_ID}_${timestamp}"

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
  export DRL_MULTI_SEED=20260724
  export DRL_MULTI_TEST_LAUNCHFILE='multi_robot_scenario_strong_interaction_pilot_5.launch'
  export DRL_MULTI_SCENARIO=manifest
  export DRL_MULTI_MANIFEST_PATH='$MANIFEST'
  export DRL_MULTI_MANIFEST_SAMPLING=cycle
  export DRL_MULTI_TEST_FILE_NAME='$MODEL_NAME'
  export DRL_MULTI_TEST_TARGET_EPISODES=30
  export DRL_MULTI_TEST_STATE_PATH='./checkpoints/${run_tag}_state.pt'
  export DRL_MULTI_TEST_STATS_PATH='./results/${run_tag}.npy'
  export DRL_MULTI_TRAJECTORY_JSONL='$trajectory_file'
  export DRL_MULTI_RECORD_RAW_LIDAR=1
  export DRL_MULTI_RAW_LIDAR_VOXEL_SIZE=0.05
  export DRL_MULTI_RAW_LIDAR_MAX_RANGE=6.0
  cd '$PROJECT_ROOT/catkin_ws'
  source devel_isolated/setup.bash
  cd '$TD3_DIR'
  python3 -u test_velodyne_td3_multi.py
" >"$log_file" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "Started 5D XYZ lidar cluster shape probe."
echo "PID: $(cat "$PID_FILE")"
echo "Scenarios: 30 fixed cases (deep/close/margin = 10/10/10)"
echo "Log: $log_file"
echo "Trajectory: $trajectory_file"
echo "Expected runtime: roughly 20-40 minutes."
