#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
RUN_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究/G11_D_Gate复核与独立准入"
RUNTIME_DIR="$RUN_DIR/local_data"
ACTIVE_LOG_DIR="$PROJECT_ROOT/logs/active/gate-g11-d2-admission"
MANIFEST="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/g11_d2_admission_v1/validation.json.gz"
BASE_MODEL="TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best"
INTERACTION_MODEL="interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016"
DETECTOR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"
A1_GATE="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究/G11_A1_当前协议时序pilot/local_data/training/seed20260804/any/T1/best.pt"
B2_GATE="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究/G11_B_student_rollout_v1/local_data/training/seed20260804/any/T1/best.pt"
OLD_G2A_GATE="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/results/06_Gate开发/D5_G2_interaction_gate_v1/local_data/model/oracle_front_v1/best.pt"
PID_FILE="$PROJECT_ROOT/.g11_d2_admission.pid"
ROS_PORT=14823
GAZEBO_PORT=14923
EPISODES=200
SEED=20260809

stop_runtime_children() {
  local pgid child_pids
  pgid="$(ps -o pgid= -p $$ | tr -d ' ')"
  child_pids="$(
    ps -eo pid=,pgid= | awk -v pgid="$pgid" -v self="$$" \
      '$2 == pgid && $1 != self { print $1 }'
  )"
  if [[ -n "$child_pids" ]]; then
    xargs -r kill 2>/dev/null <<<"$child_pids" || true
    sleep 2
    child_pids="$(
      ps -eo pid=,pgid= | awk -v pgid="$pgid" -v self="$$" \
        '$2 == pgid && $1 != self { print $1 }'
    )"
    [[ -z "$child_pids" ]] || xargs -r kill -KILL 2>/dev/null <<<"$child_pids" || true
  fi
}

cleanup() {
  stop_runtime_children
  unlink "$PID_FILE" 2>/dev/null || true
}
trap cleanup EXIT

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
export GAZEBO_RESOURCE_PATH="$PROJECT_ROOT/catkin_ws/src/multi_robot_scenario/launch"
export DRL_MULTI_NUM_AGENTS=5
export DRL_MULTI_TEST_LAUNCHFILE="multi_robot_scenario_strong_interaction_pilot_5.launch"
export DRL_MULTI_SCENARIO=manifest
export DRL_MULTI_MANIFEST_PATH="$MANIFEST"
export DRL_MULTI_MANIFEST_SAMPLING=cycle
export DRL_MULTI_TEST_TARGET_EPISODES="$EPISODES"
export DRL_MULTI_TEST_ACTOR_MODE=full
export DRL_MULTI_DENSE_ACTOR_MODE=full
export DRL_MULTI_RAW_LIDAR_VOXEL_SIZE=0.01
export DRL_MULTI_RAW_LIDAR_MAX_RANGE=6.0
export DRL_MULTI_FIXED_PHYSICS_STEP_SIZE=0.001
export DRL_MULTI_REQUIRE_FIXED_STEP_SERVICE=1
unset DRL_MULTI_CASE_ORACLE_MAP DRL_MULTI_RULE_ORACLE_MODE

mkdir -p "$ACTIVE_LOG_DIR" "$RUNTIME_DIR/results" "$RUNTIME_DIR/checkpoints"

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
  /usr/bin/python3 - "$1" "$MANIFEST" "$EPISODES" <<'PY'
import gzip
import json
import sys
import numpy as np

rows = np.load(sys.argv[1], allow_pickle=True)
episodes = int(sys.argv[3])
with gzip.open(sys.argv[2], "rt", encoding="utf-8") as handle:
    expected = {item["scenario_id"] for item in json.load(handle)["scenarios"]}
if rows.shape != (episodes, 17):
    raise SystemExit("D2 result has wrong shape: %s" % (rows.shape,))
observed = {str(item) for item in rows[:, 12]}
if observed != expected or len(observed) != episodes:
    raise SystemExit("D2 result scenario IDs do not match the manifest")
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
    raise SystemExit("D2 partial result has wrong shape: %s" % (rows.shape,))
if len(set(rows[:, 12].tolist())) != len(rows):
    raise SystemExit("D2 partial result contains duplicate scenarios")
if int(state.get("episode_num", -1)) != len(rows):
    raise SystemExit("D2 state/result episode mismatch")
manifest_state = state.get("manifest_sampling_state") or {}
if int(manifest_state.get("curriculum_case_index", -1)) != len(rows):
    raise SystemExit("D2 state/result manifest index mismatch")
print(len(rows))
PY
}

clear_policy_environment() {
  unset DRL_MULTI_DENSE_ACTOR_FILE DRL_MULTI_GATE_DETECTOR_CHECKPOINT
  unset DRL_MULTI_GATE_CHECKPOINT DRL_MULTI_GATE_SWITCH_ON_THRESHOLD
  unset DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD DRL_MULTI_GATE_MINIMUM_HOLD_STEPS
  unset DRL_MULTI_GATE_EVALUATION_STRIDE DRL_MULTI_ORACLE_INTERACTION_DISTANCE
  unset DRL_MULTI_MIN_LIDAR_SWITCH_ON_DISTANCE
  unset DRL_MULTI_MIN_LIDAR_SWITCH_OFF_DISTANCE
  unset DRL_MULTI_MIN_LIDAR_MINIMUM_HOLD_STEPS
}

configure_policy() {
  local policy="$1"
  clear_policy_environment
  export DRL_MULTI_STANDARD_ACTOR_FILE="$BASE_MODEL"
  case "$policy" in
    5a)
      export DRL_MULTI_ACTOR_SELECTION_MODE=single
      ;;
    epoch16)
      export DRL_MULTI_STANDARD_ACTOR_FILE="$INTERACTION_MODEL"
      export DRL_MULTI_ACTOR_SELECTION_MODE=single
      ;;
    rule)
      export DRL_MULTI_DENSE_ACTOR_FILE="$INTERACTION_MODEL"
      export DRL_MULTI_ACTOR_SELECTION_MODE=min_lidar_gate
      export DRL_MULTI_MIN_LIDAR_SWITCH_ON_DISTANCE=2.0
      export DRL_MULTI_MIN_LIDAR_SWITCH_OFF_DISTANCE=2.2
      export DRL_MULTI_MIN_LIDAR_MINIMUM_HOLD_STEPS=3
      ;;
    old_g2a|a1|b2)
      export DRL_MULTI_DENSE_ACTOR_FILE="$INTERACTION_MODEL"
      export DRL_MULTI_ACTOR_SELECTION_MODE=learned_gate
      export DRL_MULTI_GATE_DETECTOR_CHECKPOINT="$DETECTOR"
      if [[ "$policy" == "old_g2a" ]]; then
        export DRL_MULTI_GATE_CHECKPOINT="$OLD_G2A_GATE"
        export DRL_MULTI_GATE_SWITCH_ON_THRESHOLD=0.44
        export DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD=0.34
        export DRL_MULTI_GATE_EVALUATION_STRIDE=1
      elif [[ "$policy" == "a1" ]]; then
        export DRL_MULTI_GATE_CHECKPOINT="$A1_GATE"
        export DRL_MULTI_GATE_SWITCH_ON_THRESHOLD=0.28
        export DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD=0.18
        export DRL_MULTI_GATE_EVALUATION_STRIDE=2
      else
        export DRL_MULTI_GATE_CHECKPOINT="$B2_GATE"
        export DRL_MULTI_GATE_SWITCH_ON_THRESHOLD=0.43
        export DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD=0.33
        export DRL_MULTI_GATE_EVALUATION_STRIDE=2
      fi
      export DRL_MULTI_GATE_MINIMUM_HOLD_STEPS=3
      ;;
    oracle)
      export DRL_MULTI_DENSE_ACTOR_FILE="$INTERACTION_MODEL"
      export DRL_MULTI_ACTOR_SELECTION_MODE=interaction_oracle
      export DRL_MULTI_ORACLE_INTERACTION_DISTANCE=2.0
      ;;
    *)
      echo "Unknown D2 policy: $policy" >&2
      return 2
      ;;
  esac
}

run_one() {
  local policy="$1"
  local run_name="g11_d2_${policy}_r1_s${SEED}"
  local state_path="$RUNTIME_DIR/checkpoints/${run_name}_state.pt"
  local stats_path="$RUNTIME_DIR/results/${run_name}.npy"
  local attempt status log_file progress completed=0

  if [[ -f "$stats_path" ]] && verify_result "$stats_path" 2>/dev/null; then
    echo "Skipping completed run: $run_name"
    return 0
  fi
  export DRL_MULTI_SEED="$SEED"
  export DRL_MULTI_TEST_FILE_NAME="$run_name"
  export DRL_MULTI_TEST_STATE_PATH="$state_path"
  export DRL_MULTI_TEST_STATS_PATH="$stats_path"
  configure_policy "$policy"

  for attempt in $(seq 1 10); do
    if [[ "$attempt" -eq 1 && ! -f "$ACTIVE_LOG_DIR/${run_name}.log" ]]; then
      log_file="$ACTIVE_LOG_DIR/${run_name}.log"
    else
      log_file="$ACTIVE_LOG_DIR/${run_name}_resume${attempt}_$(date +%Y%m%d_%H%M%S).log"
    fi
    echo "Starting $run_name attempt $attempt"
    wait_for_ports
    set +e
    (cd "$TD3_DIR" && nice -n 10 python3 -u test_velodyne_td3_multi.py) \
      >"$log_file" 2>&1
    status=$?
    set -e
    stop_runtime_children
    wait_for_ports
    if verify_result "$stats_path" 2>/dev/null; then
      completed=1
      echo "Completed $run_name on attempt $attempt"
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
    echo "$run_name interrupted at $progress/$EPISODES (exit $status); restarting Gazebo"
  done
  [[ "$completed" -eq 1 ]] || {
    echo "$run_name did not complete after 10 attempts" >&2
    return 1
  }
}

run_one 5a
run_one rule
run_one old_g2a
run_one a1
run_one b2
run_one oracle
run_one epoch16

python3 "$PROJECT_ROOT/scripts/analyze_g11_d2_admission.py"
echo "G11-D2 admission complete."
