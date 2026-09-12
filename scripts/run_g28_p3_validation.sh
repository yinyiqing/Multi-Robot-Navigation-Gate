#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
G28="$BASE/28_RewardAwareGate稳健修正版"
MANIFEST="$BASE/datasets/fixed_v1/dense/validation.json.gz"
MANIFEST_SHA256="2d1dde389f927b924fa5993c47460bc60bac42aa9506ae3869c3139c9d1264b7"
CHECKPOINT="$G28/local_data/training/seed20260911_final/best.pt"
SUMMARY="$G28/local_data/training/seed20260911_final/summary.json"
RESULT_DIR="$G28/local_data/validation/results"
STATE_DIR="$G28/local_data/validation/checkpoints"
LOG_DIR="$G28/logs/p3_validation"
PID_FILE="$ROOT/.g28_p3_validation.pid"
LAUNCHFILE="$LOG_DIR/runtime_g28_p3.launch"
METHOD="${G28_P3_METHOD:-b4}"
SEED="${G28_P3_SEED:-20260911}"
if [[ "$METHOD" == "b2" ]]; then
  CHECKPOINT="$BASE/11_可部署在线Gate研究/G11_B_student_rollout_v1/local_data/training/seed20260804/any/T1/best.pt"
elif [[ "$METHOD" == "b5" ]]; then
  G29="$BASE/29_RewardAware事件证据Gate"
  CHECKPOINT="$G29/local_data/training/seed20260911_v2/best.pt"
  RESULT_DIR="$G29/local_data/validation/results"
  STATE_DIR="$G29/local_data/validation/checkpoints"
  LOG_DIR="$G29/logs/validation"
else
  [[ "$METHOD" == "b4" || "$METHOD" == "b5" ]] || { echo "Unknown reward-aware method: $METHOD" >&2; exit 2; }
fi
LOG_PREFIX="${METHOD}"
RESULT="$RESULT_DIR/g29_dense256_${METHOD}_s${SEED}.npy"
STATE="$STATE_DIR/g29_dense256_${METHOD}_s${SEED}_state.pt"
ROS_PORT=18520
GAZEBO_PORT=19520
FIVE_A="TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best"
EPOCH16="interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016"
DETECTOR="$BASE/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"

[[ -f "$CHECKPOINT" ]] || { echo "G28/B2 checkpoint is missing" >&2; exit 2; }
[[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" == "$MANIFEST_SHA256" ]] || {
  echo "Dense validation manifest hash mismatch" >&2; exit 1;
}
if [[ "$METHOD" == "b4" ]]; then
python3 - "$SUMMARY" "$CHECKPOINT" <<'PY'
import hashlib, json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
actual = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
if summary.get("checkpoint", {}).get("sha256") != actual:
    raise SystemExit("G28 checkpoint/summary hash mismatch")
PY
fi

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' <"$PID_FILE")"
  [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null && {
    echo "G28 P3 already running as PID $old_pid" >&2; exit 1;
  }
  unlink "$PID_FILE"
fi

mkdir -p "$RESULT_DIR" "$STATE_DIR" "$LOG_DIR"
echo "$$" > "$PID_FILE"
/usr/bin/python3 "$ROOT/scripts/generate_multi_robot_launch.py" --num-agents 5 --output "$LAUNCHFILE"

stop_runtime() {
  local pgid children
  pgid="$(ps -o pgid= -p $$ | tr -d ' ')"
  children="$(ps -eo pid=,pgid= | awk -v p="$pgid" -v s="$$" '$2 == p && $1 != s {print $1}')"
  [[ -z "$children" ]] || xargs -r kill -TERM 2>/dev/null <<<"$children" || true
  sleep 3
  children="$(ps -eo pid=,pgid= | awk -v p="$pgid" -v s="$$" '$2 == p && $1 != s {print $1}')"
  [[ -z "$children" ]] || xargs -r kill -KILL 2>/dev/null <<<"$children" || true
  fuser -k -KILL "${ROS_PORT}/tcp" "${GAZEBO_PORT}/tcp" >/dev/null 2>&1 || true
}
cleanup() { stop_runtime; unlink "$PID_FILE" 2>/dev/null || true; }
trap cleanup EXIT

verify_result() {
  python3 - "$RESULT" "$MANIFEST" <<'PY'
import gzip, json, sys, numpy as np
rows = np.load(sys.argv[1], allow_pickle=True)
with gzip.open(sys.argv[2], "rt", encoding="utf-8") as handle:
    expected = [str(item["scenario_id"]) for item in json.load(handle)["scenarios"][:256]]
if rows.shape != (256, 17): raise SystemExit("wrong result shape")
if [str(item) for item in rows[:, 12]] != expected: raise SystemExit("scenario order mismatch")
if sum(int(row[6]) + int(row[7]) + int(row[10]) for row in rows) != 1280:
    raise SystemExit("terminal accounting mismatch")
PY
}

verify_partial() {
  python3 - "$RESULT" "$STATE" <<'PY'
import sys, numpy as np, torch
rows = np.load(sys.argv[1], allow_pickle=True)
state = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
if rows.ndim != 2 or rows.shape[1] != 17 or not 0 < len(rows) < 256:
    raise SystemExit("invalid partial result")
if len(set(rows[:, 12].tolist())) != len(rows): raise SystemExit("duplicate scenarios")
if int(state.get("episode_num", -1)) != len(rows): raise SystemExit("state/result mismatch")
manifest_state = state.get("manifest_sampling_state") or {}
if int(manifest_state.get("curriculum_case_index", -1)) != len(rows):
    raise SystemExit("manifest state mismatch")
print(len(rows))
PY
}

if verify_result 2>/dev/null; then
  echo "G28 B4 validation already complete; reusing it"
else
  exec 9>/tmp/local_critic_multi_robot_training.lock
  flock 9
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
  export DRL_MULTI_TEST_TARGET_EPISODES=256 DRL_MULTI_FIXED_PHYSICS_STEP_SIZE=0.001
  export DRL_MULTI_REQUIRE_FIXED_STEP_SERVICE=1 DRL_MULTI_TEST_ACTOR_MODE=full DRL_MULTI_DENSE_ACTOR_MODE=full
  export DRL_MULTI_SEED="$SEED" DRL_MULTI_TEST_FILE_NAME="g28_dense256_${METHOD}_s${SEED}"
  export DRL_MULTI_TEST_STATS_PATH="$RESULT" DRL_MULTI_TEST_STATE_PATH="$STATE"
  export DRL_MULTI_STANDARD_ACTOR_FILE="$FIVE_A" DRL_MULTI_DENSE_ACTOR_FILE="$EPOCH16"
  export DRL_MULTI_ACTOR_SELECTION_MODE=learned_gate
  export DRL_MULTI_GATE_DETECTOR_CHECKPOINT="$DETECTOR" DRL_MULTI_GATE_CHECKPOINT="$CHECKPOINT"
  export DRL_MULTI_GATE_SWITCH_ON_THRESHOLD=0.43 DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD=0.33
  export DRL_MULTI_GATE_MINIMUM_HOLD_STEPS=3 DRL_MULTI_GATE_EVALUATION_STRIDE=2
  for attempt in $(seq 1 10); do
    echo "Starting G28 ${METHOD} validation attempt $attempt"
    set +e
    (cd "$ROOT/TD3" && python3 -u test_velodyne_td3_multi.py) >"$LOG_DIR/${LOG_PREFIX}_attempt${attempt}.log" 2>&1
    status=$?
    set -e
    stop_runtime
    if verify_result 2>/dev/null; then
      echo "G28 ${METHOD} validation completed"
      break
    fi
    if [[ ! -f "$RESULT" || ! -f "$STATE" ]]; then
      echo "G28 validation failed before resumable state (exit=$status)" >&2; exit 1
    fi
    progress="$(verify_partial)" || exit 1
    echo "G28 validation interrupted at $progress/256 (exit=$status); resuming exact run"
    [[ "$attempt" != 10 ]] || { echo "G28 validation failed after 10 attempts" >&2; exit 1; }
  done
fi

python3 "$ROOT/scripts/analyze_g28_p3_validation.py"
