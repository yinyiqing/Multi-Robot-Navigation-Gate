#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
MANIFEST="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/g12_r2_curriculum_v1/n5/validation.json.gz"
RUN_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/12_参数匹配单Actor容量对照/local_data/r2_n5_admission"
RESULTS_DIR="$RUN_DIR/results"
CHECKPOINT_DIR="$RUN_DIR/checkpoints"
LOG_DIR="$PROJECT_ROOT/logs/active/capacity-wide-g12-r2/n5-admission"
LAUNCHFILE="$LOG_DIR/runtime_g12_r2_n5_admission.launch"
PID_FILE="$PROJECT_ROOT/.g12_r2_n5_admission.pid"
LOCK_FILE="/tmp/local_critic_multi_robot_training.lock"
BASE_MODEL="TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best"
MODE="${1:-20k}"
case "$MODE" in
  20k)
    R2_MODEL="capacity_wide_r2_s4_broad_n5_seed20260816_best"
    R2_POLICY=r2
    SUMMARY_FILE=summary.json
    ;;
  10k)
    R2_MODEL="capacity_wide_r2_s4_broad_n5_seed20260816_epoch_001"
    R2_POLICY=r2_10k
    SUMMARY_FILE=summary_10k.json
    ;;
  *)
    echo "Unknown N5 admission mode: $MODE" >&2
    exit 2
    ;;
esac
ROS_PORT=15251
GAZEBO_PORT=15351
EPISODES=120
SEED=20260817

stop_runtime_children() {
  local pgid child_pids
  pgid="$(ps -o pgid= -p $$ | tr -d ' ')"
  child_pids="$(ps -eo pid=,pgid= | awk -v pgid="$pgid" -v self="$$" '$2 == pgid && $1 != self { print $1 }')"
  if [[ -n "$child_pids" ]]; then
    xargs -r kill -TERM 2>/dev/null <<<"$child_pids" || true
    sleep 3
    child_pids="$(ps -eo pid=,pgid= | awk -v pgid="$pgid" -v self="$$" '$2 == pgid && $1 != self { print $1 }')"
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
flock -n 9 || { echo "Multi-robot evaluation lock is busy" >&2; exit 1; }
mkdir -p "$RESULTS_DIR" "$CHECKPOINT_DIR" "$LOG_DIR"

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
export DRL_MULTI_NUM_AGENTS=5
export DRL_MULTI_SEED="$SEED"
export DRL_MULTI_TEST_LAUNCHFILE="$LAUNCHFILE"
export DRL_MULTI_TEST_ACTOR_MODE=full
export DRL_MULTI_ACTOR_SELECTION_MODE=single
export DRL_MULTI_TEST_TARGET_EPISODES="$EPISODES"
export DRL_MULTI_SCENARIO=manifest
export DRL_MULTI_MANIFEST_PATH="$MANIFEST"
export DRL_MULTI_MANIFEST_SAMPLING=cycle
export DRL_MULTI_FIXED_PHYSICS_STEP_SIZE=0.001
export DRL_MULTI_REQUIRE_FIXED_STEP_SERVICE=1
unset DRL_MULTI_DENSE_ACTOR_FILE DRL_MULTI_DENSE_ACTOR_MODE
unset DRL_MULTI_CASE_ORACLE_MAP DRL_MULTI_RULE_ORACLE_MODE

verify_result() {
  /usr/bin/python3 - "$1" "$MANIFEST" "$EPISODES" <<'PY'
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

verify_partial_result() {
  /usr/bin/python3 - "$1" "$2" "$EPISODES" <<'PY'
import sys
import numpy as np
import torch

rows = np.load(sys.argv[1], allow_pickle=True)
state = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
target = int(sys.argv[3])
if rows.ndim != 2 or rows.shape[1] != 17 or not 0 < len(rows) < target:
    raise SystemExit("wrong partial result shape: %s" % (rows.shape,))
if len(set(str(row[12]) for row in rows)) != len(rows):
    raise SystemExit("partial result contains duplicate scenario IDs")
if int(state.get("episode_num", -1)) != len(rows):
    raise SystemExit("state/result episode mismatch")
manifest_state = state.get("manifest_sampling_state") or {}
if int(manifest_state.get("curriculum_case_index", -1)) != len(rows):
    raise SystemExit("state/result manifest index mismatch")
print(len(rows))
PY
}

wait_for_ports() {
  local port
  for _ in $(seq 1 60); do
    for port in "$ROS_PORT" "$GAZEBO_PORT"; do
      if ss -ltnH | awk '{print $4}' | grep -Eq ":${port}$"; then
        break
      fi
      port=""
    done
    [[ -z "$port" ]] && return 0
    sleep 1
  done
  echo "Admission ROS/Gazebo ports did not become free" >&2
  return 1
}

run_one() {
  local policy="$1" actor="$2"
  local run_name="g12_r2_n5_admission_${policy}_s${SEED}"
  local state_path="$CHECKPOINT_DIR/${run_name}_state.pt"
  local stats_path="$RESULTS_DIR/${run_name}.npy"
  local log_file attempt status progress completed=0

  if [[ -f "$stats_path" ]] && verify_result "$stats_path" 2>/dev/null; then
    echo "Skipping completed run: $run_name"
    return
  fi
  rm -f "$state_path" "$stats_path"
  export DRL_MULTI_TEST_FILE_NAME="$run_name"
  export DRL_MULTI_STANDARD_ACTOR_FILE="$actor"
  export DRL_MULTI_TEST_STATE_PATH="$state_path"
  export DRL_MULTI_TEST_STATS_PATH="$stats_path"
  for attempt in $(seq 1 5); do
    if [[ "$attempt" -eq 1 ]]; then
      log_file="$LOG_DIR/${run_name}.log"
    else
      log_file="$LOG_DIR/${run_name}_resume${attempt}_$(date +%Y%m%d_%H%M%S).log"
    fi
    echo "Starting $run_name attempt $attempt with Actor $actor"
    wait_for_ports
    set +e
    (cd "$TD3_DIR" && nice -n 10 python3 -u test_velodyne_td3_multi.py) >"$log_file" 2>&1
    status=$?
    set -e
    stop_runtime_children
    wait_for_ports
    if verify_result "$stats_path" 2>/dev/null; then
      completed=1
      break
    fi
    if [[ ! -f "$stats_path" || ! -f "$state_path" ]]; then
      echo "$run_name failed before writing resumable state (exit $status)" >&2
      return 1
    fi
    progress="$(verify_partial_result "$stats_path" "$state_path")" || {
      echo "$run_name produced inconsistent partial state (exit $status)" >&2
      return 1
    }
    echo "$run_name interrupted at $progress/$EPISODES; restarting runtime"
  done
  [[ "$completed" -eq 1 ]] || {
    echo "$run_name did not complete after 5 attempts" >&2
    return 1
  }
  echo "Completed $run_name"
}

run_one 5a "$BASE_MODEL"
run_one "$R2_POLICY" "$R2_MODEL"

/usr/bin/python3 "$PROJECT_ROOT/scripts/analyze_g12_r2_n5_admission.py" \
  --manifest "$MANIFEST" \
  --results-dir "$RESULTS_DIR" \
  --output "$RUN_DIR/$SUMMARY_FILE" \
  --seed "$SEED" \
  --candidate-policy "$R2_POLICY" >"$LOG_DIR/analysis_${MODE}.log" 2>&1
echo "G12-R2 N5 paired admission $MODE complete."
