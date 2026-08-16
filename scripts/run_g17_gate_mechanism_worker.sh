#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$ROOT/TD3"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
RUN_DIR="$BASE/22_G17_Gate机制对照/local_data"
RESULT_DIR="$RUN_DIR/results"
STATE_DIR="$RUN_DIR/checkpoints"
MANIFEST="$BASE/datasets/fixed_v1/views/g12_full_scene_selection_v1/validation.json.gz"
LOG_DIR="$ROOT/logs/active/g17-gate-mechanism"
ARCHIVE_DIR="$ROOT/logs/archive/validation/g17_gate_mechanism"
LAUNCHFILE="$LOG_DIR/runtime_g17_gate_mechanism.launch"
PID_FILE="$ROOT/.g17_gate_mechanism.pid"
FIVE_A="TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best"
EPOCH17="avoidance_actor_from_5a_balanced_continue_e20_s20260813_best"
ROS_PORT=17623
GAZEBO_PORT=17723
SEEDS=(20260824 20260825)

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
rows = np.load(sys.argv[1], allow_pickle=True)
with gzip.open(sys.argv[2], "rt", encoding="utf-8") as f:
    ids = [str(x["scenario_id"]) for x in json.load(f)["scenarios"]]
if rows.shape != (120, 17) or [str(x[12]) for x in rows] != ids or len(set(ids)) != 120:
    raise SystemExit(1)
if sum(int(x[6]) + int(x[7]) + int(x[10]) for x in rows) != 600:
    raise SystemExit(1)
PY
}

run_one() {
  local policy="$1" seed="$2" run
  run="g17_${policy}_s${seed}"
  local result="$RESULT_DIR/${run}.npy" state="$STATE_DIR/${run}_state.pt"
  local attempt status log
  if [[ -f "$result" ]] && verify_result "$result" 2>/dev/null; then
    echo "Skipping completed $run"
    return
  fi
  export DRL_MULTI_SEED="$seed" DRL_MULTI_TEST_FILE_NAME="$run"
  export DRL_MULTI_TEST_STATS_PATH="$result" DRL_MULTI_TEST_STATE_PATH="$state"
  export DRL_MULTI_STANDARD_ACTOR_FILE="$FIVE_A" DRL_MULTI_DENSE_ACTOR_MODE=full
  unset DRL_MULTI_GATE_DETECTOR_CHECKPOINT DRL_MULTI_GATE_CHECKPOINT
  unset DRL_MULTI_GATE_SWITCH_ON_THRESHOLD DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD
  unset DRL_MULTI_GATE_MINIMUM_HOLD_STEPS DRL_MULTI_GATE_EVALUATION_STRIDE
  case "$policy" in
    epoch17_always_on)
      export DRL_MULTI_ACTOR_SELECTION_MODE=single
      export DRL_MULTI_STANDARD_ACTOR_FILE="$EPOCH17"
      unset DRL_MULTI_DENSE_ACTOR_FILE DRL_MULTI_ORACLE_INTERACTION_DISTANCE ;;
    rule_2m_privileged)
      export DRL_MULTI_ACTOR_SELECTION_MODE=interaction_oracle
      export DRL_MULTI_DENSE_ACTOR_FILE="$EPOCH17"
      export DRL_MULTI_ORACLE_INTERACTION_DISTANCE=2.0 ;;
    *) echo "Unknown policy: $policy" >&2; return 2 ;;
  esac
  for attempt in 1 2 3; do
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
    echo "$run attempt $attempt incomplete (exit=$status)" | tee -a "$LOG_DIR/runner.log"
  done
  echo "$run failed after 3 attempts" >&2
  return 1
}

exec 9>/tmp/local_critic_multi_robot_training.lock
echo "[$(date -Is)] Waiting for multi-robot lock" | tee -a "$LOG_DIR/runner.log"
flock -w 43200 9
echo "[$(date -Is)] Acquired multi-robot lock" | tee -a "$LOG_DIR/runner.log"

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

for seed in "${SEEDS[@]}"; do
  run_one epoch17_always_on "$seed"
  run_one rule_2m_privileged "$seed"
done

/usr/bin/python3 "$ROOT/scripts/analyze_g17_gate_mechanism.py" | tee "$LOG_DIR/analysis.log"
[[ ! -e "$ARCHIVE_DIR" ]] || { echo "Archive exists: $ARCHIVE_DIR" >&2; exit 1; }
mkdir -p "$(dirname "$ARCHIVE_DIR")"
cp "$LOG_DIR/runner.log" "$RUN_DIR/runner.log" 2>/dev/null || true
mv "$LOG_DIR" "$ARCHIVE_DIR"
echo "[$(date -Is)] Completed and archived at $ARCHIVE_DIR"
