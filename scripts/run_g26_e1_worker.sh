#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
MANIFEST="${G26_E1_MANIFEST:?G26_E1_MANIFEST is required}"
EPISODES="${G26_E1_EPISODES:?G26_E1_EPISODES is required}"
SEEDS_TEXT="${G26_E1_SEEDS:?G26_E1_SEEDS is required}"
LOG_DIR="${G26_E1_LOG_DIR:?G26_E1_LOG_DIR is required}"
RESULT_DIR="${G26_E1_RESULT_DIR:?G26_E1_RESULT_DIR is required}"
STATE_DIR="${G26_E1_STATE_DIR:?G26_E1_STATE_DIR is required}"
ARCHIVE_DIR="${G26_E1_ARCHIVE_DIR:?G26_E1_ARCHIVE_DIR is required}"
PID_FILE="${G26_E1_PID_FILE:?G26_E1_PID_FILE is required}"
LAUNCHFILE="${G26_E1_LAUNCHFILE:?G26_E1_LAUNCHFILE is required}"
NF_CHECKPOINT="${G26_E1_NF_CHECKPOINT:?G26_E1_NF_CHECKPOINT is required}"
ROS_PORT="${G26_E1_ROS_PORT:?G26_E1_ROS_PORT is required}"
GAZEBO_PORT="${G26_E1_GAZEBO_PORT:?G26_E1_GAZEBO_PORT is required}"

FIVE_A="TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best"
EPOCH16="interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016"
DETECTOR="$BASE/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"
B2="$BASE/11_可部署在线Gate研究/G11_B_student_rollout_v1/local_data/training/seed20260804/any/T1/best.pt"
METHODS=(5a nf_switch b2)
read -r -a SEEDS <<<"$SEEDS_TEXT"

stop_runtime_children() {
  local pgid child_pids
  pgid="$(ps -o pgid= -p $$ | tr -d ' ')"
  child_pids="$(ps -eo pid=,pgid= | awk -v p="$pgid" -v s="$$" '$2 == p && $1 != s {print $1}')"
  [[ -z "$child_pids" ]] || xargs -r kill -TERM 2>/dev/null <<<"$child_pids" || true
  sleep 2
  child_pids="$(ps -eo pid=,pgid= | awk -v p="$pgid" -v s="$$" '$2 == p && $1 != s {print $1}')"
  [[ -z "$child_pids" ]] || xargs -r kill -KILL 2>/dev/null <<<"$child_pids" || true
  fuser -k -KILL "${ROS_PORT}/tcp" "${GAZEBO_PORT}/tcp" >/dev/null 2>&1 || true
}

cleanup() {
  stop_runtime_children
  unlink "$PID_FILE" 2>/dev/null || true
}
trap cleanup EXIT

exec 9>/tmp/local_critic_multi_robot_training.lock
echo "Waiting for the single-Gazebo lock"
flock 9
echo "Acquired the single-Gazebo lock"

set +u
source /opt/ros/noetic/setup.bash
source "$ROOT/env.python.sh"
source "$ROOT/catkin_ws/devel_isolated/setup.bash"
set -u

export CUDA_VISIBLE_DEVICES=""
export ROS_HOSTNAME=localhost ROS_MASTER_URI="http://localhost:$ROS_PORT" ROS_PORT_SIM="$ROS_PORT"
export GAZEBO_MASTER_URI="http://localhost:$GAZEBO_PORT" GAZEBO_IP=127.0.0.1
export GAZEBO_RESOURCE_PATH="$ROOT/catkin_ws/src/multi_robot_scenario/launch"
export DRL_MULTI_NUM_AGENTS=5 DRL_MULTI_TEST_LAUNCHFILE="$LAUNCHFILE"
export DRL_MULTI_SCENARIO=manifest DRL_MULTI_MANIFEST_PATH="$MANIFEST" DRL_MULTI_MANIFEST_SAMPLING=cycle
export DRL_MULTI_TEST_TARGET_EPISODES="$EPISODES" DRL_MULTI_FIXED_PHYSICS_STEP_SIZE=0.001
export DRL_MULTI_REQUIRE_FIXED_STEP_SERVICE=1 DRL_MULTI_TEST_ACTOR_MODE=full DRL_MULTI_DENSE_ACTOR_MODE=full
unset DRL_MULTI_CASE_ORACLE_MAP DRL_MULTI_RULE_ORACLE_MODE

mkdir -p "$LOG_DIR" "$RESULT_DIR" "$STATE_DIR"

wait_for_ports() {
  for _ in $(seq 1 60); do
    if ! ss -ltnH | awk '{print $4}' | rg -q ":${ROS_PORT}$|:${GAZEBO_PORT}$"; then return 0; fi
    sleep 1
  done
  echo "ROS/Gazebo ports did not become free" >&2
  return 1
}

verify_result() {
  python3 - "$1" "$MANIFEST" "$EPISODES" <<'PY'
import gzip
import json
import sys
from pathlib import Path

import numpy as np

rows = np.load(sys.argv[1], allow_pickle=True)
episodes = int(sys.argv[3])
with gzip.open(sys.argv[2], "rt", encoding="utf-8") if str(sys.argv[2]).endswith(".gz") else Path(sys.argv[2]).open("r", encoding="utf-8") as handle:
    expected = [str(item["scenario_id"]) for item in json.load(handle)["scenarios"][:episodes]]
if rows.shape != (episodes, 17):
    raise SystemExit("wrong result shape")
if [str(item) for item in rows[:, 12]] != expected:
    raise SystemExit("scenario order mismatch")
if sum(int(row[6]) + int(row[7]) + int(row[10]) for row in rows) != episodes * 5:
    raise SystemExit("terminal accounting mismatch")
PY
}

verify_partial_result() {
  python3 - "$1" "$2" "$EPISODES" <<'PY'
import sys

import numpy as np
import torch

rows = np.load(sys.argv[1], allow_pickle=True)
state = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
target = int(sys.argv[3])
if rows.ndim != 2 or rows.shape[1] != 17 or not 0 < len(rows) < target:
    raise SystemExit("invalid partial result")
if len(set(rows[:, 12].tolist())) != len(rows):
    raise SystemExit("duplicate scenarios")
if int(state.get("episode_num", -1)) != len(rows):
    raise SystemExit("state/result mismatch")
manifest_state = state.get("manifest_sampling_state") or {}
if int(manifest_state.get("curriculum_case_index", -1)) != len(rows):
    raise SystemExit("manifest state mismatch")
print(len(rows))
PY
}

clear_policy_environment() {
  unset DRL_MULTI_STANDARD_ACTOR_FILE DRL_MULTI_DENSE_ACTOR_FILE DRL_MULTI_ACTOR_SELECTION_MODE
  unset DRL_MULTI_GATE_DETECTOR_CHECKPOINT DRL_MULTI_GATE_CHECKPOINT
  unset DRL_MULTI_GATE_SWITCH_ON_THRESHOLD DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD
  unset DRL_MULTI_GATE_MINIMUM_HOLD_STEPS DRL_MULTI_GATE_EVALUATION_STRIDE
  unset DRL_MULTI_NF_SWITCH_CHECKPOINT
}

configure_policy() {
  local method="$1"
  clear_policy_environment
  export DRL_MULTI_STANDARD_ACTOR_FILE="$FIVE_A" DRL_MULTI_ACTOR_SELECTION_MODE=single
  case "$method" in
    5a) ;;
    nf_switch)
      export DRL_MULTI_DENSE_ACTOR_FILE="$EPOCH16" DRL_MULTI_ACTOR_SELECTION_MODE=normalizing_flow_switch
      export DRL_MULTI_NF_SWITCH_CHECKPOINT="$NF_CHECKPOINT" ;;
    b2)
      export DRL_MULTI_DENSE_ACTOR_FILE="$EPOCH16" DRL_MULTI_ACTOR_SELECTION_MODE=learned_gate
      export DRL_MULTI_GATE_DETECTOR_CHECKPOINT="$DETECTOR" DRL_MULTI_GATE_CHECKPOINT="$B2"
      export DRL_MULTI_GATE_SWITCH_ON_THRESHOLD=0.43 DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD=0.33
      export DRL_MULTI_GATE_MINIMUM_HOLD_STEPS=3 DRL_MULTI_GATE_EVALUATION_STRIDE=2 ;;
    *) echo "Unknown E1 method: $method" >&2; return 2 ;;
  esac
}

run_one() {
  local method="$1" seed="$2" run_name result state
  local attempt status log progress
  run_name="g26_e1_${method}_s${seed}"
  result="$RESULT_DIR/${run_name}.npy"
  state="$STATE_DIR/${run_name}_state.pt"
  if [[ -f "$result" ]] && verify_result "$result" 2>/dev/null; then
    echo "Skipping completed $run_name"
    return 0
  fi
  export DRL_MULTI_SEED="$seed" DRL_MULTI_TEST_FILE_NAME="$run_name"
  export DRL_MULTI_TEST_STATS_PATH="$result" DRL_MULTI_TEST_STATE_PATH="$state"
  configure_policy "$method"
  for attempt in $(seq 1 10); do
    log="$LOG_DIR/${run_name}_attempt${attempt}.log"
    echo "Starting $run_name attempt $attempt"
    wait_for_ports
    set +e
    (cd "$ROOT/TD3" && nice -n 10 python3 -u test_velodyne_td3_multi.py) >"$log" 2>&1
    status=$?
    set -e
    stop_runtime_children
    wait_for_ports
    if [[ -f "$result" ]] && verify_result "$result" 2>/dev/null; then
      echo "Completed $run_name"
      return 0
    fi
    if [[ ! -f "$result" || ! -f "$state" ]]; then
      echo "$run_name failed before writing resumable state (exit=$status)" >&2
      return 1
    fi
    progress="$(verify_partial_result "$result" "$state")" || return 1
    echo "$run_name interrupted at $progress/$EPISODES (exit=$status); restarting"
  done
  echo "$run_name failed after 10 attempts" >&2
  return 1
}

for seed_index in "${!SEEDS[@]}"; do
  seed="${SEEDS[$seed_index]}"
  for method_index in "${!METHODS[@]}"; do
    run_one "${METHODS[$method_index]}" "$seed"
  done
done

python3 - "$MANIFEST" "$EPISODES" "$SEEDS_TEXT" "$RESULT_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest, episodes, seeds, result_dir = sys.argv[1:]
path = Path(manifest)
record = {
    "status": "complete",
    "manifest": manifest,
    "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "episodes_per_method": int(episodes),
    "seeds": [int(value) for value in seeds.split()],
    "methods": ["5a", "nf_switch", "b2"],
    "total_episodes": int(episodes) * len(seeds.split()) * 3,
    "sealed_test_read": True,
    "actor_or_router_updated": False,
}
output = Path(result_dir).parent / "e1_completion.json"
output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(output)
PY

[[ ! -e "$ARCHIVE_DIR" ]] || { echo "Archive exists: $ARCHIVE_DIR" >&2; exit 1; }
mkdir -p "$(dirname "$ARCHIVE_DIR")"
mv "$LOG_DIR" "$ARCHIVE_DIR"
echo "G26-E1 evaluation complete: 3 methods x ${EPISODES} scenes x ${#SEEDS[@]} repeats"
