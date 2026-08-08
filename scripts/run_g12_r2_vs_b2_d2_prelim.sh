#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
RUN_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/12_参数匹配单Actor容量对照/local_data/r2_vs_b2_d2_prelim"
RESULTS_DIR="$RUN_DIR/results"
CHECKPOINT_DIR="$RUN_DIR/checkpoints"
LOG_DIR="$PROJECT_ROOT/logs/active/capacity-wide-g12-r2-vs-b2-d2"
MANIFEST="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/g11_d2_admission_v1/validation.json.gz"
ACTOR="capacity_wide_r2_s4_broad_n5_seed20260816_epoch_001"
RESULT="$RESULTS_DIR/g12_r2_10k_d2_r1_s20260809.npy"
STATE="$CHECKPOINT_DIR/g12_r2_10k_d2_r1_s20260809_state.pt"
LOG="$LOG_DIR/g12_r2_10k_d2_r1_s20260809.log"
ROS_PORT=15523
GAZEBO_PORT=15623
EPISODES=200
SEED=20260809
LOCK_FILE=/tmp/local_critic_multi_robot_training.lock

exec 9>"$LOCK_FILE"
flock -n 9 || { echo "Multi-robot evaluation lock is busy" >&2; exit 1; }
mkdir -p "$RESULTS_DIR" "$CHECKPOINT_DIR" "$LOG_DIR"

set +u
source /opt/ros/noetic/setup.bash
source "$PROJECT_ROOT/env.python.sh"
source "$PROJECT_ROOT/catkin_ws/devel_isolated/setup.bash"
set -u

export CUDA_VISIBLE_DEVICES=""
export ROS_HOSTNAME=localhost
export ROS_MASTER_URI="http://localhost:$ROS_PORT"
export ROS_PORT_SIM="$ROS_PORT"
export GAZEBO_MASTER_URI="http://localhost:$GAZEBO_PORT"
export GAZEBO_IP=127.0.0.1
export GAZEBO_RESOURCE_PATH="$PROJECT_ROOT/catkin_ws/src/multi_robot_scenario/launch"
export DRL_MULTI_NUM_AGENTS=5
export DRL_MULTI_SEED="$SEED"
export DRL_MULTI_TEST_FILE_NAME=g12_r2_10k_d2_r1_s20260809
export DRL_MULTI_TEST_LAUNCHFILE=multi_robot_scenario_strong_interaction_pilot_5.launch
export DRL_MULTI_TEST_ACTOR_MODE=full
export DRL_MULTI_ACTOR_SELECTION_MODE=single
export DRL_MULTI_TEST_TARGET_EPISODES="$EPISODES"
export DRL_MULTI_SCENARIO=manifest
export DRL_MULTI_MANIFEST_PATH="$MANIFEST"
export DRL_MULTI_MANIFEST_SAMPLING=cycle
export DRL_MULTI_FIXED_PHYSICS_STEP_SIZE=0.001
export DRL_MULTI_REQUIRE_FIXED_STEP_SERVICE=1
export DRL_MULTI_STANDARD_ACTOR_FILE="$ACTOR"
export DRL_MULTI_TEST_STATE_PATH="$STATE"
export DRL_MULTI_TEST_STATS_PATH="$RESULT"
unset DRL_MULTI_DENSE_ACTOR_FILE DRL_MULTI_DENSE_ACTOR_MODE
unset DRL_MULTI_CASE_ORACLE_MAP DRL_MULTI_RULE_ORACLE_MODE

wait_for_ports() {
  for _ in $(seq 1 60); do
    if ! ss -ltnH | awk '{print $4}' | rg -q ":${ROS_PORT}$|:${GAZEBO_PORT}$"; then
      return 0
    fi
    sleep 1
  done
  echo "ROS/Gazebo ports did not become free" >&2
  return 1
}

verify_result() {
  /usr/bin/python3 - "$RESULT" "$MANIFEST" "$EPISODES" <<'PY'
import gzip
import json
import sys
import numpy as np

rows = np.load(sys.argv[1], allow_pickle=True)
with gzip.open(sys.argv[2], "rt", encoding="utf-8") as handle:
    expected = [str(item["scenario_id"]) for item in json.load(handle)["scenarios"]]
observed = [str(row[12]) for row in rows]
if rows.shape != (int(sys.argv[3]), 17):
    raise SystemExit("wrong result shape: %s" % (rows.shape,))
if observed != expected or len(set(observed)) != len(observed):
    raise SystemExit("result scenario IDs do not match manifest order")
PY
}

cleanup() {
  local pgid child_pids
  pgid="$(ps -o pgid= -p $$ | tr -d ' ')"
  child_pids="$(ps -eo pid=,pgid= | awk -v pgid="$pgid" -v self="$$" '$2 == pgid && $1 != self { print $1 }')"
  [[ -z "$child_pids" ]] || xargs -r kill -TERM <<<"$child_pids" 2>/dev/null || true
  sleep 2
  child_pids="$(ps -eo pid=,pgid= | awk -v pgid="$pgid" -v self="$$" '$2 == pgid && $1 != self { print $1 }')"
  [[ -z "$child_pids" ]] || xargs -r kill -KILL <<<"$child_pids" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if verify_result 2>/dev/null; then
  echo "R2 D2 preliminary result already complete: $RESULT"
else
  rm -f "$RESULT"
  wait_for_ports
  set +e
  (cd "$TD3_DIR" && nice -n 10 python3 -u test_velodyne_td3_multi.py) >"$LOG" 2>&1
  status=$?
  set -e
  wait_for_ports
  verify_result || {
    echo "R2 D2 preliminary evaluation failed with exit $status" >&2
    exit "$status"
  }
fi

B2_RESULT="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究/G11_D_Gate复核与独立准入/local_data/results/g11_d2_b2_r1_s20260809.npy"
python3 "$PROJECT_ROOT/scripts/compare_actor_validation.py" \
  "$B2_RESULT" "$RESULT" \
  --baseline-label b2 \
  --candidate-label r2_10k \
  --manifest "$MANIFEST" \
  --output "$RUN_DIR/summary.json"
echo "R2-10k vs B2 D2 preliminary comparison complete."
