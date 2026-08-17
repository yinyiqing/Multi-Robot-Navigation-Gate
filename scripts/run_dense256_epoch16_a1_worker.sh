#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
RUN_DIR="$BASE/24_Dense256_epoch16_A1复测/local_data"
RESULT="$RUN_DIR/results/dense256_epoch16_a1_s20260810.npy"
STATE="$RUN_DIR/checkpoints/dense256_epoch16_a1_s20260810_state.pt"
MANIFEST="$BASE/datasets/fixed_v1/dense/validation.json.gz"
LOG_DIR="$ROOT/logs/active/dense256-epoch16-a1"
ARCHIVE_DIR="$ROOT/logs/archive/validation/dense256_epoch16_a1"
LAUNCHFILE="$LOG_DIR/runtime_dense256_epoch16_a1.launch"
PID_FILE="$ROOT/.dense256_epoch16_a1.pid"
ROS_PORT=18223
GAZEBO_PORT=18323

mkdir -p "$LOG_DIR" "$(dirname "$RESULT")" "$(dirname "$STATE")"
stop_runtime() {
  local pgid children
  pgid="$(ps -o pgid= -p $$ | tr -d ' ')"
  children="$(ps -eo pid=,pgid= | awk -v p="$pgid" -v s="$$" '$2==p && $1!=s {print $1}')"
  [[ -z "$children" ]] || xargs -r kill -TERM 2>/dev/null <<<"$children" || true
  sleep 3
  children="$(ps -eo pid=,pgid= | awk -v p="$pgid" -v s="$$" '$2==p && $1!=s {print $1}')"
  [[ -z "$children" ]] || xargs -r kill -KILL 2>/dev/null <<<"$children" || true
  fuser -k -KILL "${ROS_PORT}/tcp" "${GAZEBO_PORT}/tcp" >/dev/null 2>&1 || true
}
cleanup() { stop_runtime; unlink "$PID_FILE" 2>/dev/null || true; }
trap cleanup EXIT
verify_result() {
  /usr/bin/python3 - "$RESULT" "$MANIFEST" <<'PY'
import gzip,json,sys,numpy as np
x=np.load(sys.argv[1],allow_pickle=True)
with gzip.open(sys.argv[2],"rt",encoding="utf-8") as f: ids=[str(v["scenario_id"]) for v in json.load(f)["scenarios"][:256]]
if x.shape!=(256,17) or [str(v) for v in x[:,12]]!=ids or len(set(ids))!=256: raise SystemExit(1)
if sum(int(v[6])+int(v[7])+int(v[10]) for v in x)!=1280: raise SystemExit(1)
PY
}
if [[ -f "$RESULT" ]] && verify_result 2>/dev/null; then echo "Result already complete"; exit 0; fi
exec 9>/tmp/local_critic_multi_robot_training.lock
echo "[$(date -Is)] Waiting for multi-robot lock"
flock -w 43200 9
echo "[$(date -Is)] Acquired multi-robot lock"
set +u
source /opt/ros/noetic/setup.bash
source "$ROOT/env.python.sh"
source "$ROOT/catkin_ws/devel_isolated/setup.bash"
set -u
/usr/bin/python3 "$ROOT/scripts/generate_multi_robot_launch.py" --num-agents 5 --output "$LAUNCHFILE"
export CUDA_VISIBLE_DEVICES=""
export ROS_HOSTNAME=localhost ROS_MASTER_URI="http://localhost:$ROS_PORT" ROS_PORT_SIM="$ROS_PORT"
export GAZEBO_MASTER_URI="http://localhost:$GAZEBO_PORT"
export GAZEBO_RESOURCE_PATH="$ROOT/catkin_ws/src/multi_robot_scenario/launch"
export DRL_MULTI_NUM_AGENTS=5 DRL_MULTI_TEST_LAUNCHFILE="$LAUNCHFILE"
export DRL_MULTI_SCENARIO=manifest DRL_MULTI_MANIFEST_PATH="$MANIFEST" DRL_MULTI_MANIFEST_SAMPLING=cycle
export DRL_MULTI_TEST_TARGET_EPISODES=256 DRL_MULTI_FIXED_PHYSICS_STEP_SIZE=0.001
export DRL_MULTI_REQUIRE_FIXED_STEP_SERVICE=1 DRL_MULTI_TEST_ACTOR_MODE=full
export DRL_MULTI_SEED=20260810 DRL_MULTI_TEST_FILE_NAME=dense256_epoch16_a1_s20260810
export DRL_MULTI_TEST_STATS_PATH="$RESULT" DRL_MULTI_TEST_STATE_PATH="$STATE"
export DRL_MULTI_STANDARD_ACTOR_FILE=TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best
export DRL_MULTI_DENSE_ACTOR_FILE=interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016
export DRL_MULTI_DENSE_ACTOR_MODE=full DRL_MULTI_ACTOR_SELECTION_MODE=learned_gate
export DRL_MULTI_GATE_DETECTOR_CHECKPOINT="$BASE/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"
export DRL_MULTI_GATE_CHECKPOINT="$BASE/11_可部署在线Gate研究/G11_A1_当前协议时序pilot/local_data/training/seed20260804/any/T1/best.pt"
export DRL_MULTI_GATE_SWITCH_ON_THRESHOLD=0.28 DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD=0.18
export DRL_MULTI_GATE_MINIMUM_HOLD_STEPS=3 DRL_MULTI_GATE_EVALUATION_STRIDE=2
for attempt in 1 2 3 4 5; do
  echo "Starting attempt $attempt"
  set +e
  (cd "$ROOT/TD3" && nice -n 10 python3 -u test_velodyne_td3_multi.py) >"$LOG_DIR/dense256_epoch16_a1_attempt${attempt}.log" 2>&1
  status=$?
  set -e
  stop_runtime
  if [[ -f "$RESULT" ]] && verify_result 2>/dev/null; then
    /usr/bin/python3 "$ROOT/scripts/analyze_dense256_epoch16_a1.py" | tee "$LOG_DIR/analysis.log"
    mkdir -p "$(dirname "$ARCHIVE_DIR")"
    [[ ! -e "$ARCHIVE_DIR" ]] || { echo "Archive exists" >&2; exit 1; }
    mv "$LOG_DIR" "$ARCHIVE_DIR"
    echo "Completed and archived: $ARCHIVE_DIR"
    exit 0
  fi
  echo "Attempt $attempt incomplete (exit=$status)"
done
echo "Failed after 5 attempts" >&2
exit 1
