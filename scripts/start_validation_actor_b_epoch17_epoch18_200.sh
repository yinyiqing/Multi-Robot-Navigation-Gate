#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
EXPERIMENT_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/09_局部专家全程化/04_epoch17_epoch18固定200场复核"
LOCAL_DATA_DIR="$EXPERIMENT_DIR/local_data"
MANIFEST_PATH="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/dense_validation_monitor_v1/validation.json.gz"
LAUNCHFILE="multi_robot_scenario_strong_interaction_pilot_5.launch"
BASE_ACTOR="actor_b_from_epoch16_full_pilot_v1_s20260802_epoch_017"
CANDIDATE_ACTOR="actor_b_from_epoch16_full_pilot_v1_s20260802_epoch_018"
SEED="${DRL_MULTI_ACTOR_B_VALIDATION_SEED:-20260803}"
BASE_ROS_PORT="${DRL_MULTI_ACTOR_B_BASE_ROS_PORT:-14211}"
BASE_GAZEBO_PORT="${DRL_MULTI_ACTOR_B_BASE_GAZEBO_PORT:-14311}"
CANDIDATE_ROS_PORT="${DRL_MULTI_ACTOR_B_CANDIDATE_ROS_PORT:-14212}"
CANDIDATE_GAZEBO_PORT="${DRL_MULTI_ACTOR_B_CANDIDATE_GAZEBO_PORT:-14312}"
PID_FILE="$PROJECT_ROOT/.validation_actor_b_epoch17_epoch18_200.pid"
LOCK_FILE="/tmp/local_critic_multi_robot_training.lock"

[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "Seed must be an integer"; exit 2; }
[[ -f "$MANIFEST_PATH" ]] || { echo "Manifest is missing: $MANIFEST_PATH"; exit 1; }
[[ -f "$TD3_DIR/assets/$LAUNCHFILE" ]] || { echo "Launch file is missing: $LAUNCHFILE"; exit 1; }
for actor in "$BASE_ACTOR" "$CANDIDATE_ACTOR"; do
  [[ -f "$TD3_DIR/pytorch_models/${actor}_actor.pth" ]] || {
    echo "Actor is missing: $TD3_DIR/pytorch_models/${actor}_actor.pth"
    exit 1
  }
done

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Actor B paired validation is already running with PID $old_pid"
    exit 1
  fi
  unlink "$PID_FILE"
fi

for port in \
  "$BASE_ROS_PORT" "$BASE_GAZEBO_PORT" \
  "$CANDIDATE_ROS_PORT" "$CANDIDATE_GAZEBO_PORT"; do
  if ss -ltnH | awk '{print $4}' | rg -q ":${port}$"; then
    echo "Port $port is already in use"
    exit 1
  fi
done

mkdir -p "$LOCAL_DATA_DIR/logs" "$LOCAL_DATA_DIR/stats" "$LOCAL_DATA_DIR/state"
for label in epoch17 epoch18; do
  for path in \
    "$LOCAL_DATA_DIR/stats/${label}.npy" \
    "$LOCAL_DATA_DIR/state/${label}.pt"; do
    [[ ! -e "$path" ]] || {
      echo "Output already exists: $path"
      echo "Move the completed result or remove an incomplete local_data run first."
      exit 1
    }
  done
done

timestamp="$(date +%Y%m%d_%H%M%S)"
runner_log="$LOCAL_DATA_DIR/logs/paired_${timestamp}_runner.log"
base_log="$LOCAL_DATA_DIR/logs/epoch17_${timestamp}.log"
candidate_log="$LOCAL_DATA_DIR/logs/epoch18_${timestamp}.log"

setsid bash -lc "
  set -eo pipefail
  exec 9>'$LOCK_FILE'
  flock -n 9 || { echo 'Multi-robot training/validation lock is busy'; exit 1; }

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
  export GAZEBO_RESOURCE_PATH='$PROJECT_ROOT/catkin_ws/src/multi_robot_scenario/launch'
  export DRL_MULTI_NUM_AGENTS=5
  export DRL_MULTI_SEED='$SEED'
  export DRL_MULTI_TEST_LAUNCHFILE='$LAUNCHFILE'
  export DRL_MULTI_ACTOR_SELECTION_MODE=single
  export DRL_MULTI_TEST_ACTOR_MODE=full
  export DRL_MULTI_TEST_TARGET_EPISODES=200
  export DRL_MULTI_SCENARIO=manifest
  export DRL_MULTI_MANIFEST_PATH='$MANIFEST_PATH'
  export DRL_MULTI_MANIFEST_SAMPLING=cycle
  unset DRL_MULTI_DENSE_ACTOR_FILE DRL_MULTI_DENSE_ACTOR_MODE
  unset DRL_MULTI_CASE_ORACLE_MAP DRL_MULTI_RULE_ORACLE_MODE
  cd '$TD3_DIR'

  export ROS_MASTER_URI='http://localhost:$BASE_ROS_PORT'
  export ROS_PORT_SIM='$BASE_ROS_PORT'
  export GAZEBO_MASTER_URI='http://localhost:$BASE_GAZEBO_PORT'
  export DRL_MULTI_TEST_FILE_NAME='actor_b_epoch17_fixed200_s$SEED'
  export DRL_MULTI_STANDARD_ACTOR_FILE='$BASE_ACTOR'
  export DRL_MULTI_TEST_STATE_PATH='$LOCAL_DATA_DIR/state/epoch17.pt'
  export DRL_MULTI_TEST_STATS_PATH='$LOCAL_DATA_DIR/stats/epoch17.npy'
  python3 -u test_velodyne_td3_multi.py >'$base_log' 2>&1

  sleep 5

  export ROS_MASTER_URI='http://localhost:$CANDIDATE_ROS_PORT'
  export ROS_PORT_SIM='$CANDIDATE_ROS_PORT'
  export GAZEBO_MASTER_URI='http://localhost:$CANDIDATE_GAZEBO_PORT'
  export DRL_MULTI_TEST_FILE_NAME='actor_b_epoch18_fixed200_s$SEED'
  export DRL_MULTI_STANDARD_ACTOR_FILE='$CANDIDATE_ACTOR'
  export DRL_MULTI_TEST_STATE_PATH='$LOCAL_DATA_DIR/state/epoch18.pt'
  export DRL_MULTI_TEST_STATS_PATH='$LOCAL_DATA_DIR/stats/epoch18.npy'
  python3 -u test_velodyne_td3_multi.py >'$candidate_log' 2>&1
" >"$runner_log" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "Started Actor B epoch17 -> epoch18 fixed 200-scene paired validation."
echo "PID: $(cat "$PID_FILE")"
echo "Seed: $SEED"
echo "Manifest: $MANIFEST_PATH"
echo "Epoch 17 log: $base_log"
echo "Epoch 18 log: $candidate_log"
echo "Runner log: $runner_log"
