#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
RUN_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究/G11_C_50场闭环pilot"
RUNTIME_DIR="$RUN_DIR/local_data"
ACTIVE_LOG_DIR="$PROJECT_ROOT/local/logs/gate-g11-c-pilot"
MANIFEST="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/g11_c_pilot_v1/validation.json.gz"
BASE_MODEL="TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best"
INTERACTION_MODEL="interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016"
DETECTOR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"
A1_GATE="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究/G11_A1_当前协议时序pilot/local_data/training/seed20260804/any/T1/best.pt"
B2_GATE="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究/G11_B_student_rollout_v1/local_data/training/seed20260804/any/T1/best.pt"
PID_FILE="$PROJECT_ROOT/.g11_c_pilot.pid"
ROS_PORT=14623
GAZEBO_PORT=14723

stop_runtime_children() {
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
    if [[ -n "$child_pids" ]]; then
      xargs -r kill -KILL 2>/dev/null <<<"$child_pids" || true
    fi
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
export DRL_MULTI_TEST_TARGET_EPISODES=50
export DRL_MULTI_STANDARD_ACTOR_FILE="$BASE_MODEL"
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
  local stats_path="$1"
  python3 - "$stats_path" <<'PY'
import sys
import numpy as np

rows = np.load(sys.argv[1], allow_pickle=True)
if rows.shape != (50, 17):
    raise SystemExit("pilot result has wrong shape: %s" % (rows.shape,))
if len(set(rows[:, 12].tolist())) != 50:
    raise SystemExit("pilot result does not contain 50 unique scenarios")
PY
}

verify_partial_result() {
  local stats_path="$1"
  local state_path="$2"
  python3 - "$stats_path" "$state_path" <<'PY'
import sys
import numpy as np
import torch

rows = np.load(sys.argv[1], allow_pickle=True)
state = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
if rows.ndim != 2 or rows.shape[1] != 17 or not 0 < len(rows) < 50:
    raise SystemExit("pilot partial result has wrong shape: %s" % (rows.shape,))
if len(set(rows[:, 12].tolist())) != len(rows):
    raise SystemExit("pilot partial result contains duplicate scenarios")
if int(state.get("episode_num", -1)) != len(rows):
    raise SystemExit("pilot state/result episode mismatch")
manifest_state = state.get("manifest_sampling_state") or {}
if int(manifest_state.get("curriculum_case_index", -1)) != len(rows):
    raise SystemExit("pilot state/result manifest index mismatch")
print(len(rows))
PY
}

run_one() {
  local policy="$1"
  local repeat="$2"
  local seed="$3"
  local run_name="g11_c_${policy}_r${repeat}_s${seed}"
  local state_path="$RUNTIME_DIR/checkpoints/${run_name}_state.pt"
  local stats_path="$RUNTIME_DIR/results/${run_name}.npy"

  if [[ -f "$stats_path" ]] && verify_result "$stats_path" 2>/dev/null; then
    echo "Skipping completed run: $run_name"
    return 0
  fi

  export DRL_MULTI_SEED="$seed"
  export DRL_MULTI_TEST_FILE_NAME="$run_name"
  export DRL_MULTI_TEST_STATE_PATH="$state_path"
  export DRL_MULTI_TEST_STATS_PATH="$stats_path"
  case "$policy" in
    5a)
      export DRL_MULTI_ACTOR_SELECTION_MODE=single
      unset DRL_MULTI_DENSE_ACTOR_FILE DRL_MULTI_GATE_DETECTOR_CHECKPOINT
      unset DRL_MULTI_GATE_CHECKPOINT DRL_MULTI_GATE_SWITCH_ON_THRESHOLD
      unset DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD DRL_MULTI_GATE_MINIMUM_HOLD_STEPS
      unset DRL_MULTI_GATE_EVALUATION_STRIDE
      ;;
    a1)
      export DRL_MULTI_DENSE_ACTOR_FILE="$INTERACTION_MODEL"
      export DRL_MULTI_ACTOR_SELECTION_MODE=learned_gate
      export DRL_MULTI_GATE_DETECTOR_CHECKPOINT="$DETECTOR"
      export DRL_MULTI_GATE_CHECKPOINT="$A1_GATE"
      export DRL_MULTI_GATE_SWITCH_ON_THRESHOLD=0.28
      export DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD=0.18
      export DRL_MULTI_GATE_MINIMUM_HOLD_STEPS=3
      export DRL_MULTI_GATE_EVALUATION_STRIDE=2
      ;;
    b2)
      export DRL_MULTI_DENSE_ACTOR_FILE="$INTERACTION_MODEL"
      export DRL_MULTI_ACTOR_SELECTION_MODE=learned_gate
      export DRL_MULTI_GATE_DETECTOR_CHECKPOINT="$DETECTOR"
      export DRL_MULTI_GATE_CHECKPOINT="$B2_GATE"
      export DRL_MULTI_GATE_SWITCH_ON_THRESHOLD=0.43
      export DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD=0.33
      export DRL_MULTI_GATE_MINIMUM_HOLD_STEPS=3
      export DRL_MULTI_GATE_EVALUATION_STRIDE=2
      ;;
    *)
      echo "Unknown pilot policy: $policy" >&2
      return 2
      ;;
  esac

  local attempt status log_file completed=0
  for attempt in $(seq 1 5); do
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
    echo "$run_name interrupted at $progress/50 (exit $status); restarting Gazebo"
  done
  if [[ "$completed" -ne 1 ]]; then
    echo "$run_name did not complete after 5 attempts" >&2
    return 1
  fi
}

run_one 5a 1 20260805
run_one a1 1 20260805
run_one b2 1 20260805
run_one b2 2 20260806
run_one a1 2 20260806
run_one 5a 2 20260806

python3 "$PROJECT_ROOT/scripts/analyze_g11_c_pilot.py"
echo "G11-C pilot complete."
