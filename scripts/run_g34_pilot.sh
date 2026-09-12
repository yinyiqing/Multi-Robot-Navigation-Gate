#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
G34="$BASE/34_双头RewardAwareGate"
MANIFEST="${G34_MANIFEST:-$BASE/datasets/fixed_v1/dense/validation.json.gz}"
MANIFEST_SHA256="${G34_MANIFEST_SHA256:-2d1dde389f927b924fa5993c47460bc60bac42aa9506ae3869c3139c9d1264b7}"
CHECKPOINT="$G34/local_data/training/seed20260912/best_runtime.pt"
AUDIT="$G34/local_data/training/seed20260912/checkpoint_audit.json"
STAGE="${G34_STAGE:-pilot32}"
RUN_NAME="${G34_RUN_NAME:-g34_dense32}"
RESULT_DIR="$G34/local_data/$STAGE/results"
STATE_DIR="$G34/local_data/$STAGE/checkpoints"
LOG_DIR="$G34/logs/$STAGE"
PID_FILE="${G34_PID_FILE:-$ROOT/.g34_pilot.pid}"
LAUNCHFILE="$LOG_DIR/runtime_${STAGE}.launch"
ANALYZER="${G34_ANALYZER:-$ROOT/scripts/analyze_g34_pilot.py}"
SEED="${G34_SEED:-20260912}"
TARGET_EPISODES="${G34_TARGET_EPISODES:-32}"
ROS_PORT="${G34_ROS_PORT:-18640}"
GAZEBO_PORT="${G34_GAZEBO_PORT:-19640}"
FIVE_A="TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best"
EPOCH16="interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016"
DETECTOR="$BASE/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"

[[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" == "$MANIFEST_SHA256" ]] || {
  echo "Dense validation manifest hash mismatch" >&2
  exit 1
}

mkdir -p "$RESULT_DIR" "$STATE_DIR" "$LOG_DIR"
source "$ROOT/env.python.sh"
python3 "$ROOT/scripts/audit_g34_checkpoint.py" >"$LOG_DIR/checkpoint_audit.log"
python3 - "$AUDIT" <<'PY'
import json, sys
audit = json.load(open(sys.argv[1], encoding="utf-8"))
if not audit.get("passed"):
    raise SystemExit("G34 checkpoint audit did not pass")
PY

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' <"$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "G34 ${STAGE} is already running as PID $old_pid" >&2
    exit 1
  fi
  unlink "$PID_FILE"
fi
echo "$$" >"$PID_FILE"
/usr/bin/python3 "$ROOT/scripts/generate_multi_robot_launch.py" --num-agents 5 --output "$LAUNCHFILE"

stop_runtime() {
  local pgid children
  pgid="$(ps -o pgid= -p $$ | tr -d ' ')"
  children="$(ps -eo pid=,pgid= | awk -v p="$pgid" -v s="$$" '$2 == p && $1 != s {print $1}')"
  for child in $children; do kill -TERM "$child" 2>/dev/null || true; done
  children="$(ps -eo pid=,pgid= | awk -v p="$pgid" -v s="$$" '$2 == p && $1 != s {print $1}')"
  for child in $children; do kill -KILL "$child" 2>/dev/null || true; done
  fuser -k -KILL "${ROS_PORT}/tcp" "${GAZEBO_PORT}/tcp" >/dev/null 2>&1 || true
}
cleanup() {
  stop_runtime
  unlink "$PID_FILE" 2>/dev/null || true
}
trap cleanup EXIT

verify_result() {
  local result="$1"
  python3 - "$result" "$MANIFEST" "$TARGET_EPISODES" <<'PY'
import gzip, json, sys, numpy as np
rows = np.load(sys.argv[1], allow_pickle=True)
target = int(sys.argv[3])
with gzip.open(sys.argv[2], "rt", encoding="utf-8") as handle:
    expected = [str(item["scenario_id"]) for item in json.load(handle)["scenarios"][:target]]
if rows.shape != (target, 17): raise SystemExit("wrong result shape")
if [str(item) for item in rows[:, 12]] != expected: raise SystemExit("scenario order mismatch")
if sum(int(row[6]) + int(row[7]) + int(row[10]) for row in rows) != target * 5:
    raise SystemExit("terminal accounting mismatch")
PY
}

verify_partial() {
  local result="$1" state="$2"
  python3 - "$result" "$state" "$TARGET_EPISODES" <<'PY'
import sys, numpy as np, torch
rows = np.load(sys.argv[1], allow_pickle=True)
state = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
target = int(sys.argv[3])
if rows.ndim != 2 or rows.shape[1] != 17 or not 0 < len(rows) < target:
    raise SystemExit("invalid partial result")
if len(set(rows[:, 12].tolist())) != len(rows): raise SystemExit("duplicate scenarios")
if int(state.get("episode_num", -1)) not in (len(rows), len(rows) + 1):
    raise SystemExit("state/result mismatch")
manifest_state = state.get("manifest_sampling_state") or {}
if int(manifest_state.get("curriculum_case_index", -1)) not in (len(rows), len(rows) + 1):
    raise SystemExit("manifest state mismatch")
print(len(rows))
PY
}

run_method() {
  local method="$1"
  local result="$RESULT_DIR/${RUN_NAME}_${method}_s${SEED}.npy"
  local state="$STATE_DIR/${RUN_NAME}_${method}_s${SEED}_state.pt"
  if verify_result "$result" 2>/dev/null; then
    echo "G34 ${STAGE} ${method}: existing audited result reused"
    return
  fi
  for attempt in $(seq 1 6); do
    echo "G34 ${STAGE} ${method}: attempt ${attempt}"
    set +e
    (
      set -euo pipefail
      set +u
      source /opt/ros/noetic/setup.bash
      source "$ROOT/env.python.sh"
      source "$ROOT/catkin_ws/devel_isolated/setup.bash"
      set -u
      export CUDA_VISIBLE_DEVICES="" ROS_HOSTNAME=localhost
      export ROS_MASTER_URI="http://localhost:$ROS_PORT" ROS_PORT_SIM="$ROS_PORT"
      export GAZEBO_MASTER_URI="http://localhost:$GAZEBO_PORT" GAZEBO_IP=127.0.0.1
      export GAZEBO_RESOURCE_PATH="$ROOT/catkin_ws/src/multi_robot_scenario/launch"
      export DRL_MULTI_NUM_AGENTS=5 DRL_MULTI_TEST_LAUNCHFILE="$LAUNCHFILE"
      export DRL_MULTI_SCENARIO=manifest DRL_MULTI_MANIFEST_PATH="$MANIFEST" DRL_MULTI_MANIFEST_SAMPLING=cycle
      export DRL_MULTI_TEST_TARGET_EPISODES="$TARGET_EPISODES" DRL_MULTI_FIXED_PHYSICS_STEP_SIZE=0.001
      export DRL_MULTI_REQUIRE_FIXED_STEP_SERVICE=1 DRL_MULTI_TEST_ACTOR_MODE=full
      export DRL_MULTI_SEED="$SEED" DRL_MULTI_TEST_FILE_NAME="${RUN_NAME}_${method}_s${SEED}"
      export DRL_MULTI_TEST_STATS_PATH="$result" DRL_MULTI_TEST_STATE_PATH="$state"
      export DRL_MULTI_STANDARD_ACTOR_FILE="$FIVE_A"
      if [[ "$method" == "joint" ]]; then
        export DRL_MULTI_DENSE_ACTOR_FILE="$EPOCH16" DRL_MULTI_DENSE_ACTOR_MODE=full
        export DRL_MULTI_ACTOR_SELECTION_MODE=learned_gate
        export DRL_MULTI_GATE_DETECTOR_CHECKPOINT="$DETECTOR" DRL_MULTI_GATE_CHECKPOINT="$CHECKPOINT"
        export DRL_MULTI_GATE_SWITCH_ON_THRESHOLD=0.43 DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD=0.33
        export DRL_MULTI_GATE_MINIMUM_HOLD_STEPS=3 DRL_MULTI_GATE_EVALUATION_STRIDE=2
      else
        export DRL_MULTI_DENSE_ACTOR_FILE="" DRL_MULTI_ACTOR_SELECTION_MODE=single
        unset DRL_MULTI_GATE_DETECTOR_CHECKPOINT DRL_MULTI_GATE_CHECKPOINT
      fi
      cd "$ROOT/TD3"
      python3 -u test_velodyne_td3_multi.py
    ) >"$LOG_DIR/${method}_attempt${attempt}.log" 2>&1
    status=$?
    set -e
    stop_runtime
    if verify_result "$result" 2>/dev/null; then
      echo "G34 ${STAGE} ${method}: completed and audited"
      return
    fi
    if [[ ! -f "$result" || ! -f "$state" ]]; then
      echo "G34 ${STAGE} ${method}: failed before resumable state (exit=$status)" >&2
      exit 1
    fi
    progress="$(verify_partial "$result" "$state")" || exit 1
    echo "G34 ${STAGE} ${method}: interrupted at ${progress}/${TARGET_EPISODES}; exact resume"
  done
  echo "G34 ${STAGE} ${method}: failed after six attempts" >&2
  exit 1
}

exec 9>/tmp/local_critic_multi_robot_training.lock
flock 9
run_method joint
run_method 5a
python3 "$ANALYZER" >"$LOG_DIR/analysis.log"
echo "G34 ${STAGE} queue complete"
