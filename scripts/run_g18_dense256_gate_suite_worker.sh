#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
RUN_DIR="$BASE/18_dense256当前方法复测/local_data"
RESULT_DIR="$RUN_DIR/results"
STATE_DIR="$RUN_DIR/checkpoints"
MANIFEST="$BASE/datasets/fixed_v1/dense/validation.json.gz"
LOG_DIR="$ROOT/logs/active/g18-dense256-gate-suite"
ARCHIVE_DIR="$ROOT/logs/archive/validation/g18_dense256_gate_suite"
R2B_ARCHIVE="$ROOT/logs/archive/validation/g18_dense256_r2b"
R2B_PID="$ROOT/.g18_dense256_r2b.pid"
LAUNCHFILE="$LOG_DIR/runtime_g18_dense256_gate_suite.launch"
PID_FILE="$ROOT/.g18_dense256_gate_suite.pid"
FIVE_A="TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best"
EPOCH17="avoidance_actor_from_5a_balanced_continue_e20_s20260813_best"
DETECTOR="$BASE/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"
F_A1="$BASE/11_可部署在线Gate研究/G11_F_epoch17_gate_v1/local_data/a1_training/seed20260804/any/T1/best.pt"
F_B2="$BASE/11_可部署在线Gate研究/G11_F_epoch17_gate_v1/local_data/aggregated_training/seed20260804/any/T1/best.pt"
OLD_B2="$BASE/11_可部署在线Gate研究/G11_B_student_rollout_v1/local_data/training/seed20260804/any/T1/best.pt"
ROS_PORT=17423
GAZEBO_PORT=17523

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

mkdir -p "$LOG_DIR" "$RESULT_DIR" "$STATE_DIR"
echo "Waiting for G18 R2B-best to complete"
while [[ ! -d "$R2B_ARCHIVE" ]]; do
  if [[ ! -f "$R2B_PID" ]]; then
    echo "G18 R2B-best exited without a completed archive" >&2
    exit 1
  fi
  sleep 15
done

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
export GAZEBO_MASTER_URI="http://localhost:$GAZEBO_PORT"
export GAZEBO_RESOURCE_PATH="$ROOT/catkin_ws/src/multi_robot_scenario/launch"
export DRL_MULTI_NUM_AGENTS=5 DRL_MULTI_TEST_LAUNCHFILE="$LAUNCHFILE"
export DRL_MULTI_SCENARIO=manifest DRL_MULTI_MANIFEST_PATH="$MANIFEST" DRL_MULTI_MANIFEST_SAMPLING=cycle
export DRL_MULTI_TEST_TARGET_EPISODES=256 DRL_MULTI_FIXED_PHYSICS_STEP_SIZE=0.001
export DRL_MULTI_REQUIRE_FIXED_STEP_SERVICE=1 DRL_MULTI_TEST_ACTOR_MODE=full
export DRL_MULTI_DENSE_ACTOR_MODE=full DRL_MULTI_SEED=20260810

verify_result() {
  python3 - "$1" "$MANIFEST" <<'PY'
import gzip,json,sys,numpy as np
rows=np.load(sys.argv[1],allow_pickle=True)
with gzip.open(sys.argv[2],'rt',encoding='utf-8') as f:
    ids=[str(x['scenario_id']) for x in json.load(f)['scenarios'][:256]]
if rows.shape != (256,17): raise SystemExit(1)
if [str(x[12]) for x in rows] != ids: raise SystemExit(1)
if sum(int(x[6])+int(x[7])+int(x[10]) for x in rows) != 1280: raise SystemExit(1)
PY
}

run_one() {
  local policy="$1" run="g18_dense256_${1}_s20260810"
  local result="$RESULT_DIR/${run}.npy" state="$STATE_DIR/${run}_state.pt"
  local attempt status log
  if [[ -f "$result" ]] && verify_result "$result" 2>/dev/null; then
    echo "Skipping completed $run"
    return
  fi
  export DRL_MULTI_TEST_FILE_NAME="$run" DRL_MULTI_TEST_STATS_PATH="$result"
  export DRL_MULTI_TEST_STATE_PATH="$state" DRL_MULTI_STANDARD_ACTOR_FILE="$FIVE_A"
  unset DRL_MULTI_ORACLE_INTERACTION_DISTANCE DRL_MULTI_GATE_DETECTOR_CHECKPOINT
  unset DRL_MULTI_GATE_CHECKPOINT DRL_MULTI_GATE_SWITCH_ON_THRESHOLD
  unset DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD DRL_MULTI_GATE_MINIMUM_HOLD_STEPS
  unset DRL_MULTI_GATE_EVALUATION_STRIDE DRL_MULTI_DENSE_ACTOR_FILE
  case "$policy" in
    5a)
      export DRL_MULTI_ACTOR_SELECTION_MODE=single ;;
    f_a1)
      export DRL_MULTI_ACTOR_SELECTION_MODE=learned_gate DRL_MULTI_DENSE_ACTOR_FILE="$EPOCH17"
      export DRL_MULTI_GATE_DETECTOR_CHECKPOINT="$DETECTOR" DRL_MULTI_GATE_CHECKPOINT="$F_A1"
      export DRL_MULTI_GATE_SWITCH_ON_THRESHOLD=0.29 DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD=0.19
      export DRL_MULTI_GATE_MINIMUM_HOLD_STEPS=3 DRL_MULTI_GATE_EVALUATION_STRIDE=2 ;;
    f_b2)
      export DRL_MULTI_ACTOR_SELECTION_MODE=learned_gate DRL_MULTI_DENSE_ACTOR_FILE="$EPOCH17"
      export DRL_MULTI_GATE_DETECTOR_CHECKPOINT="$DETECTOR" DRL_MULTI_GATE_CHECKPOINT="$F_B2"
      export DRL_MULTI_GATE_SWITCH_ON_THRESHOLD=0.43 DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD=0.33
      export DRL_MULTI_GATE_MINIMUM_HOLD_STEPS=3 DRL_MULTI_GATE_EVALUATION_STRIDE=2 ;;
    old_b2)
      export DRL_MULTI_ACTOR_SELECTION_MODE=learned_gate DRL_MULTI_DENSE_ACTOR_FILE="$EPOCH17"
      export DRL_MULTI_GATE_DETECTOR_CHECKPOINT="$DETECTOR" DRL_MULTI_GATE_CHECKPOINT="$OLD_B2"
      export DRL_MULTI_GATE_SWITCH_ON_THRESHOLD=0.43 DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD=0.33
      export DRL_MULTI_GATE_MINIMUM_HOLD_STEPS=3 DRL_MULTI_GATE_EVALUATION_STRIDE=2 ;;
    rule_2m)
      export DRL_MULTI_ACTOR_SELECTION_MODE=interaction_oracle DRL_MULTI_DENSE_ACTOR_FILE="$EPOCH17"
      export DRL_MULTI_ORACLE_INTERACTION_DISTANCE=2.0 ;;
    *) echo "Unknown policy: $policy" >&2; return 2 ;;
  esac
  for attempt in 1 2 3; do
    log="$LOG_DIR/${run}_attempt${attempt}.log"
    echo "Starting $run attempt $attempt"
    set +e
    (cd "$ROOT/TD3" && nice -n 10 python3 -u test_velodyne_td3_multi.py) >"$log" 2>&1
    status=$?
    set -e
    stop_runtime
    if [[ -f "$result" ]] && verify_result "$result" 2>/dev/null; then
      echo "Completed $run"
      return
    fi
    echo "$run attempt $attempt incomplete (exit=$status)"
  done
  echo "$run failed after 3 attempts" >&2
  return 1
}

run_one 5a
run_one f_a1
run_one f_b2
run_one old_b2
run_one rule_2m
python3 "$ROOT/scripts/analyze_g18_dense256_current_suite.py" >"$LOG_DIR/analysis.log" 2>&1
[[ ! -e "$ARCHIVE_DIR" ]] || { echo "Archive exists: $ARCHIVE_DIR" >&2; exit 1; }
mkdir -p "$(dirname "$ARCHIVE_DIR")"
mv "$LOG_DIR" "$ARCHIVE_DIR"
