#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
RUN_DIR="$BASE/18_dense256当前方法复测/local_data"
RESULT="$RUN_DIR/results/g18_dense256_r2b_best_s20260810.npy"
STATE="$RUN_DIR/checkpoints/g18_dense256_r2b_best_s20260810_state.pt"
MANIFEST="$BASE/datasets/fixed_v1/dense/validation.json.gz"
LOG_DIR="$ROOT/logs/active/g18-dense256-r2b"
ARCHIVE_DIR="$ROOT/logs/archive/validation/g18_dense256_r2b"
LAUNCHFILE="$LOG_DIR/runtime_g18_dense256.launch"
PID_FILE="$ROOT/.g18_dense256_r2b.pid"
MODEL="capacity_wide_r2b_5a_recipe_n5_seed20260823_best"
ROS_PORT=17223
GAZEBO_PORT=17323

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

mkdir -p "$LOG_DIR" "$RUN_DIR/results" "$RUN_DIR/checkpoints"
exec 9>/tmp/local_critic_multi_robot_training.lock
echo "Waiting for the single-Gazebo lock"
flock 9
echo "Acquired the single-Gazebo lock"

set +u
source /opt/ros/noetic/setup.bash
source "$ROOT/env.python.sh"
source "$ROOT/catkin_ws/devel_isolated/setup.bash"
set -u

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

export CUDA_VISIBLE_DEVICES=""
export ROS_HOSTNAME=localhost ROS_MASTER_URI="http://localhost:$ROS_PORT" ROS_PORT_SIM="$ROS_PORT"
export GAZEBO_MASTER_URI="http://localhost:$GAZEBO_PORT"
export GAZEBO_RESOURCE_PATH="$ROOT/catkin_ws/src/multi_robot_scenario/launch"
export DRL_MULTI_NUM_AGENTS=5 DRL_MULTI_TEST_LAUNCHFILE="$LAUNCHFILE"
export DRL_MULTI_SCENARIO=manifest DRL_MULTI_MANIFEST_PATH="$MANIFEST" DRL_MULTI_MANIFEST_SAMPLING=cycle
export DRL_MULTI_TEST_TARGET_EPISODES=256 DRL_MULTI_FIXED_PHYSICS_STEP_SIZE=0.001
export DRL_MULTI_REQUIRE_FIXED_STEP_SERVICE=1 DRL_MULTI_TEST_ACTOR_MODE=full
export DRL_MULTI_SEED=20260810 DRL_MULTI_TEST_FILE_NAME=g18_dense256_r2b_best_s20260810
export DRL_MULTI_TEST_STATS_PATH="$RESULT" DRL_MULTI_TEST_STATE_PATH="$STATE"
export DRL_MULTI_STANDARD_ACTOR_FILE="$MODEL" DRL_MULTI_ACTOR_SELECTION_MODE=single
unset DRL_MULTI_DENSE_ACTOR_FILE DRL_MULTI_GATE_DETECTOR_CHECKPOINT DRL_MULTI_GATE_CHECKPOINT

if [[ ! -f "$RESULT" ]] || ! verify_result "$RESULT" 2>/dev/null; then
  for attempt in 1 2 3; do
    echo "Starting R2B-best dense256 attempt $attempt"
    set +e
    (cd "$ROOT/TD3" && nice -n 10 python3 -u test_velodyne_td3_multi.py) \
      >"$LOG_DIR/g18_dense256_r2b_best_s20260810_attempt${attempt}.log" 2>&1
    status=$?
    set -e
    stop_runtime
    if [[ -f "$RESULT" ]] && verify_result "$RESULT" 2>/dev/null; then
      echo "Completed R2B-best dense256"
      break
    fi
    echo "Attempt $attempt incomplete (exit=$status)"
  done
fi

verify_result "$RESULT"
python3 "$ROOT/scripts/analyze_g18_dense256_r2b.py" >"$LOG_DIR/analysis.log" 2>&1
[[ ! -e "$ARCHIVE_DIR" ]] || { echo "Archive exists: $ARCHIVE_DIR" >&2; exit 1; }
mkdir -p "$(dirname "$ARCHIVE_DIR")"
mv "$LOG_DIR" "$ARCHIVE_DIR"
