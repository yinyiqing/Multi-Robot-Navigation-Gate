#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
MANIFEST="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/g12_r2_curriculum_v1/n5/validation.json.gz"
RUN_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/15_E2恢复Actor诊断与训练/local_data/recovery_oracle_epoch16_pilot"
RESULTS_DIR="$RUN_DIR/results"
CHECKPOINT_DIR="$RUN_DIR/checkpoints"
TRAJECTORY_DIR="$RUN_DIR/trajectories"
LOG_DIR="$PROJECT_ROOT/logs/active/current-generalist-r2style/e2-recovery-oracle-epoch16-pilot"
LAUNCHFILE="$LOG_DIR/runtime_e2_recovery_oracle_epoch16_pilot.launch"
PID_FILE="$PROJECT_ROOT/.e2_recovery_oracle_epoch16_pilot.pid"
LOCK_FILE="/tmp/local_critic_multi_robot_training.lock"
E2_MODEL="current_generalist_n5_efficiency_e2_s20260810_best"
INTERACTION_MODEL="interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016"
REFERENCE_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/12_参数匹配单Actor容量对照/local_data/r2_n5_admission/results"
N5_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/13_当前场景普通Actor重训/local_data/n5_admission/results"
E2_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/13_当前场景普通Actor重训/local_data/n5_efficiency_e2_admission/results"
ROS_PORT=15457
GAZEBO_PORT=15557
EPISODES=120
SEED=20260818

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
mkdir -p "$RESULTS_DIR" "$CHECKPOINT_DIR" "$TRAJECTORY_DIR" "$LOG_DIR"

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
export DRL_MULTI_ACTOR_SELECTION_MODE=recovery_oracle
export DRL_MULTI_TEST_TARGET_EPISODES="$EPISODES"
export DRL_MULTI_SCENARIO=manifest
export DRL_MULTI_MANIFEST_PATH="$MANIFEST"
export DRL_MULTI_MANIFEST_SAMPLING=cycle
export DRL_MULTI_FIXED_PHYSICS_STEP_SIZE=0.001
export DRL_MULTI_REQUIRE_FIXED_STEP_SERVICE=1
export DRL_MULTI_STANDARD_ACTOR_FILE="$E2_MODEL"
export DRL_MULTI_DENSE_ACTOR_FILE="$INTERACTION_MODEL"
export DRL_MULTI_DENSE_ACTOR_MODE=full
export DRL_MULTI_RECOVERY_ORACLE_CANDIDATE_DISTANCE=2.0
export DRL_MULTI_RECOVERY_ORACLE_RELEASE_DISTANCE=2.4
export DRL_MULTI_RECOVERY_ORACLE_PROGRESS_THRESHOLD=0.003
export DRL_MULTI_RECOVERY_ORACLE_PROGRESS_WINDOW=5
export DRL_MULTI_RECOVERY_ORACLE_DISTANCE_DELTA_THRESHOLD=0.02
export DRL_MULTI_RECOVERY_ORACLE_GOAL_DISTANCE=0.45
export DRL_MULTI_RECOVERY_ORACLE_MINIMUM_HOLD_STEPS=3
export DRL_MULTI_RECOVERY_ORACLE_MAXIMUM_HOLD_STEPS=20
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
  echo "Recovery-oracle ROS/Gazebo ports did not become free" >&2
  return 1
}

run_recovery_oracle() {
  local run_name="e2_recovery_oracle_epoch16_pilot_s${SEED}"
  local state_path="$CHECKPOINT_DIR/${run_name}_state.pt"
  local stats_path="$RESULTS_DIR/${run_name}.npy"
  local trajectory_path="$TRAJECTORY_DIR/${run_name}.jsonl"
  local log_file attempt status progress completed=0

  if [[ -f "$stats_path" ]] && verify_result "$stats_path" 2>/dev/null; then
    echo "Skipping completed run: $run_name"
    return
  fi
  rm -f "$state_path" "$stats_path" "$trajectory_path"
  export DRL_MULTI_TEST_FILE_NAME="$run_name"
  export DRL_MULTI_TEST_STATE_PATH="$state_path"
  export DRL_MULTI_TEST_STATS_PATH="$stats_path"
  export DRL_MULTI_TRAJECTORY_JSONL="$trajectory_path"
  for attempt in $(seq 1 5); do
    if [[ "$attempt" -eq 1 ]]; then
      log_file="$LOG_DIR/${run_name}.log"
    else
      log_file="$LOG_DIR/${run_name}_resume${attempt}_$(date +%Y%m%d_%H%M%S).log"
    fi
    echo "Starting $run_name attempt $attempt with standard=$E2_MODEL interaction=$INTERACTION_MODEL"
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

run_recovery_oracle

/usr/bin/python3 "$PROJECT_ROOT/scripts/analyze_current_generalist_e2_oracle_admission.py" \
  --manifest "$MANIFEST" \
  --five-a-result "$REFERENCE_DIR/g12_r2_n5_admission_5a_s20260817.npy" \
  --r2-result "$REFERENCE_DIR/g12_r2_n5_admission_r2_10k_s20260817.npy" \
  --n5-result "$N5_DIR/current_generalist_n5_admission_n5_s20260817.npy" \
  --e2-result "$E2_DIR/current_generalist_n5_efficiency_e2_admission_s20260817.npy" \
  --e2-oracle-result "$RESULTS_DIR/e2_recovery_oracle_epoch16_pilot_s${SEED}.npy" \
  --output "$RUN_DIR/summary.json" \
  --seed "$SEED" \
  --candidate-key "e2_recovery_oracle_epoch16" \
  --experiment-name "e2-recovery-oracle-epoch16-pilot" \
  --candidate-description "recovery_oracle: standard=E2, interaction=epoch16 under near-active-robot + low-progress/stagnation rule" \
  >"$LOG_DIR/analysis.log" 2>&1
echo "E2 recovery-oracle epoch16 pilot complete."
