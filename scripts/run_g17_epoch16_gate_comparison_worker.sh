#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$ROOT/TD3"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
RUN_DIR="$BASE/23_G17_epoch16同场复测/local_data"
RESULT_DIR="$RUN_DIR/results"
STATE_DIR="$RUN_DIR/checkpoints"
MANIFEST="$BASE/datasets/fixed_v1/views/g12_full_scene_selection_v1/validation.json.gz"
LOG_DIR="$ROOT/logs/active/g17-epoch16-gate-comparison"
ARCHIVE_DIR="$ROOT/logs/archive/validation/g17_epoch16_gate_comparison"
LAUNCHFILE="$LOG_DIR/runtime_g17_epoch16_comparison.launch"
PID_FILE="$ROOT/.g17_epoch16_gate_comparison.pid"
FIVE_A="TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best"
EPOCH16="interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016"
DETECTOR="$BASE/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"
A1_GATE="$BASE/11_可部署在线Gate研究/G11_A1_当前协议时序pilot/local_data/training/seed20260804/any/T1/best.pt"
B2_GATE="$BASE/11_可部署在线Gate研究/G11_B_student_rollout_v1/local_data/training/seed20260804/any/T1/best.pt"
ROS_PORT=18023
GAZEBO_PORT=18123

mkdir -p "$LOG_DIR" "$RESULT_DIR" "$STATE_DIR"
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
  /usr/bin/python3 - "$1" "$MANIFEST" <<'PY'
import gzip, json, sys
import numpy as np
rows=np.load(sys.argv[1],allow_pickle=True)
with gzip.open(sys.argv[2],"rt",encoding="utf-8") as f:
    ids=[str(x["scenario_id"]) for x in json.load(f)["scenarios"]]
if rows.shape != (120,17) or [str(x[12]) for x in rows] != ids or len(set(ids)) != 120:
    raise SystemExit(1)
if sum(int(x[6])+int(x[7])+int(x[10]) for x in rows) != 600:
    raise SystemExit(1)
PY
}

run_one() {
  local policy="$1" seed="$2" run result state attempt status log
  run="g17_epoch16_${policy}_s${seed}"
  result="$RESULT_DIR/${run}.npy"
  state="$STATE_DIR/${run}_state.pt"
  if [[ -f "$result" ]] && verify_result "$result" 2>/dev/null; then
    echo "Skipping completed $run"
    return
  fi
  export DRL_MULTI_SEED="$seed" DRL_MULTI_TEST_FILE_NAME="$run"
  export DRL_MULTI_TEST_STATS_PATH="$result" DRL_MULTI_TEST_STATE_PATH="$state"
  export DRL_MULTI_STANDARD_ACTOR_FILE="$FIVE_A" DRL_MULTI_DENSE_ACTOR_FILE="$EPOCH16"
  export DRL_MULTI_DENSE_ACTOR_MODE=full
  unset DRL_MULTI_GATE_DETECTOR_CHECKPOINT DRL_MULTI_GATE_CHECKPOINT
  unset DRL_MULTI_GATE_SWITCH_ON_THRESHOLD DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD
  unset DRL_MULTI_GATE_MINIMUM_HOLD_STEPS DRL_MULTI_GATE_EVALUATION_STRIDE
  unset DRL_MULTI_ORACLE_INTERACTION_DISTANCE
  case "$policy" in
    a1)
      export DRL_MULTI_ACTOR_SELECTION_MODE=learned_gate
      export DRL_MULTI_GATE_DETECTOR_CHECKPOINT="$DETECTOR" DRL_MULTI_GATE_CHECKPOINT="$A1_GATE"
      export DRL_MULTI_GATE_SWITCH_ON_THRESHOLD=0.28 DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD=0.18
      export DRL_MULTI_GATE_MINIMUM_HOLD_STEPS=3 DRL_MULTI_GATE_EVALUATION_STRIDE=2 ;;
    b2)
      export DRL_MULTI_ACTOR_SELECTION_MODE=learned_gate
      export DRL_MULTI_GATE_DETECTOR_CHECKPOINT="$DETECTOR" DRL_MULTI_GATE_CHECKPOINT="$B2_GATE"
      export DRL_MULTI_GATE_SWITCH_ON_THRESHOLD=0.43 DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD=0.33
      export DRL_MULTI_GATE_MINIMUM_HOLD_STEPS=3 DRL_MULTI_GATE_EVALUATION_STRIDE=2 ;;
    rule_2m)
      export DRL_MULTI_ACTOR_SELECTION_MODE=interaction_oracle
      export DRL_MULTI_ORACLE_INTERACTION_DISTANCE=2.0 ;;
    *) echo "Unknown policy: $policy" >&2; return 2 ;;
  esac
  for attempt in 1 2 3 4 5; do
    log="$LOG_DIR/${run}_attempt${attempt}.log"
    echo "Starting $run attempt $attempt"
    set +e
    (cd "$TD3_DIR" && nice -n 10 python3 -u test_velodyne_td3_multi.py) >"$log" 2>&1
    status=$?
    set -e
    stop_runtime
    if [[ -f "$result" ]] && verify_result "$result" 2>/dev/null; then
      echo "Completed $run"
      return
    fi
    echo "$run attempt $attempt incomplete (exit=$status)"
  done
  echo "$run failed after 5 attempts" >&2
  return 1
}

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
export DRL_MULTI_TEST_TARGET_EPISODES=120 DRL_MULTI_FIXED_PHYSICS_STEP_SIZE=0.001
export DRL_MULTI_REQUIRE_FIXED_STEP_SERVICE=1 DRL_MULTI_TEST_ACTOR_MODE=full

run_one a1 20260824
run_one b2 20260824
run_one rule_2m 20260824
run_one rule_2m 20260825
run_one b2 20260825
run_one a1 20260825
/usr/bin/python3 "$ROOT/scripts/analyze_g17_epoch16_gate_comparison.py" | tee "$LOG_DIR/analysis.log"
[[ ! -e "$ARCHIVE_DIR" ]] || { echo "Archive exists: $ARCHIVE_DIR" >&2; exit 1; }
mkdir -p "$(dirname "$ARCHIVE_DIR")"
cp "$LOG_DIR/runner.log" "$RUN_DIR/runner.log" 2>/dev/null || true
mv "$LOG_DIR" "$ARCHIVE_DIR"
echo "[$(date -Is)] Completed and archived at $ARCHIVE_DIR"
