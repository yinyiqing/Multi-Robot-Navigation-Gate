#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$ROOT/TD3"
BASE_DIR="$ROOT/experiments/03_保留专门化/02_论文主线"
MANIFEST="$BASE_DIR/datasets/fixed_v1/views/g12_full_scene_selection_v1/validation.json.gz"
RUN_DIR="$BASE_DIR/16_避障Actor续训/local_data/matched_admission"
RESULT_DIR="$RUN_DIR/results"
STATE_DIR="$RUN_DIR/checkpoints"
LOG_DIR="$ROOT/logs/active/avoidance-actor-matched-admission"
ARCHIVE_DIR="$ROOT/logs/archive/validation/avoidance_actor_matched_admission"
LAUNCHFILE="$LOG_DIR/runtime_avoidance_actor_matched_admission.launch"
PID_FILE="$ROOT/.avoidance_actor_matched_admission.pid"
LOCK_FILE="/tmp/local_critic_multi_robot_training.lock"
FIVE_A="TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best"
OLD_ACTOR="interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016"
NEW_ACTOR="avoidance_actor_from_5a_balanced_continue_e20_s20260813_best"
ROS_PORT=15813
GAZEBO_PORT=15913
EPISODES=120
SEEDS=(20260814 20260815)

stop_runtime() {
  local launch_pids
  launch_pids="$(
    ps -eo pid=,args= | awk -v launch="$LAUNCHFILE" \
      'index($0, "roslaunch") && index($0, launch) { print $1 }'
  )"
  if [[ -n "$launch_pids" ]]; then
    xargs -r kill -TERM 2>/dev/null <<<"$launch_pids" || true
    sleep 3
  fi
  fuser -k -TERM "${ROS_PORT}/tcp" "${GAZEBO_PORT}/tcp" >/dev/null 2>&1 || true
  sleep 3
  launch_pids="$(
    ps -eo pid=,args= | awk -v launch="$LAUNCHFILE" \
      'index($0, "roslaunch") && index($0, launch) { print $1 }'
  )"
  [[ -z "$launch_pids" ]] || xargs -r kill -KILL 2>/dev/null <<<"$launch_pids" || true
  fuser -k -KILL "${ROS_PORT}/tcp" "${GAZEBO_PORT}/tcp" >/dev/null 2>&1 || true
}

cleanup() {
  stop_runtime
  unlink "$PID_FILE" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

exec 9>"$LOCK_FILE"
flock -n 9 || { echo "Multi-robot evaluation lock is busy" >&2; exit 1; }
mkdir -p "$RESULT_DIR" "$STATE_DIR" "$LOG_DIR"
set +u
source /opt/ros/noetic/setup.bash
source "$ROOT/env.python.sh"
source "$ROOT/catkin_ws/devel_isolated/setup.bash"
set -u

export CUDA_VISIBLE_DEVICES=0
export ROS_HOSTNAME=localhost
export GAZEBO_IP=127.0.0.1
export ROS_MASTER_URI="http://localhost:$ROS_PORT"
export ROS_PORT_SIM="$ROS_PORT"
export GAZEBO_MASTER_URI="http://localhost:$GAZEBO_PORT"
export GAZEBO_RESOURCE_PATH="$ROOT/catkin_ws/src/multi_robot_scenario/launch"
export DRL_MULTI_NUM_AGENTS=5
export DRL_MULTI_TEST_LAUNCHFILE="$LAUNCHFILE"
export DRL_MULTI_TEST_ACTOR_MODE=full
export DRL_MULTI_ACTOR_SELECTION_MODE=interaction_oracle
export DRL_MULTI_TEST_TARGET_EPISODES="$EPISODES"
export DRL_MULTI_SCENARIO=manifest
export DRL_MULTI_MANIFEST_PATH="$MANIFEST"
export DRL_MULTI_MANIFEST_SAMPLING=cycle
export DRL_MULTI_FIXED_PHYSICS_STEP_SIZE=0.001
export DRL_MULTI_REQUIRE_FIXED_STEP_SERVICE=1
export DRL_MULTI_STANDARD_ACTOR_FILE="$FIVE_A"
export DRL_MULTI_DENSE_ACTOR_MODE=full
export DRL_MULTI_ORACLE_INTERACTION_DISTANCE=2.0
unset DRL_MULTI_CASE_ORACLE_MAP DRL_MULTI_RULE_ORACLE_MODE

verify_result() {
  /usr/bin/python3 - "$1" "$MANIFEST" "$EPISODES" <<'PY'
import gzip, json, sys
import numpy as np
rows=np.load(sys.argv[1],allow_pickle=True)
with gzip.open(sys.argv[2],"rt",encoding="utf-8") as f:
    expected=[str(x["scenario_id"]) for x in json.load(f)["scenarios"]]
observed=[str(x[12]) for x in rows]
if rows.shape != (int(sys.argv[3]),17) or observed != expected or len(set(observed)) != len(observed):
    raise SystemExit("result audit failed")
if sum(int(x[6])+int(x[7])+int(x[10]) for x in rows) != len(rows)*5:
    raise SystemExit("terminal accounting failed")
PY
}

run_policy() {
  local label="$1" actor="$2" seed="$3"
  local run_name="avoidance_${label}_s${seed}"
  local stats="$RESULT_DIR/${run_name}.npy" state="$STATE_DIR/${run_name}_state.pt"
  local attempt log_file status
  if [[ -f "$stats" ]] && verify_result "$stats" 2>/dev/null; then
    echo "Skipping completed $run_name"
    return
  fi
  [[ ! -f "$stats" ]] || unlink "$stats"
  [[ ! -f "$state" ]] || unlink "$state"
  export DRL_MULTI_SEED="$seed"
  export DRL_MULTI_DENSE_ACTOR_FILE="$actor"
  export DRL_MULTI_TEST_FILE_NAME="$run_name"
  export DRL_MULTI_TEST_STATE_PATH="$state"
  export DRL_MULTI_TEST_STATS_PATH="$stats"
  for attempt in 1 2 3; do
    log_file="$LOG_DIR/${run_name}_attempt${attempt}.log"
    echo "Starting $run_name attempt $attempt actor=$actor"
    set +e
    (cd "$TD3_DIR" && nice -n 10 python3 -u test_velodyne_td3_multi.py) >"$log_file" 2>&1
    status=$?
    set -e
    stop_runtime
    if [[ -f "$stats" ]] && verify_result "$stats" 2>/dev/null; then
      echo "Completed $run_name"
      return
    fi
    echo "$run_name attempt $attempt incomplete (exit=$status)"
  done
  echo "$run_name failed after 3 attempts" >&2
  return 1
}

for seed in "${SEEDS[@]}"; do
  run_policy old_e16 "$OLD_ACTOR" "$seed"
  run_policy candidate_e17 "$NEW_ACTOR" "$seed"
done

/usr/bin/python3 "$ROOT/scripts/analyze_avoidance_actor_matched_admission.py" \
  --manifest "$MANIFEST" --result-dir "$RESULT_DIR" \
  --seeds "${SEEDS[@]}" --output "$RUN_DIR/summary.json" >"$LOG_DIR/analysis.log" 2>&1

[[ ! -e "$ARCHIVE_DIR" ]] || { echo "Archive already exists: $ARCHIVE_DIR" >&2; exit 1; }
mkdir -p "$(dirname "$ARCHIVE_DIR")"
mv "$LOG_DIR" "$ARCHIVE_DIR"
echo "Avoidance Actor matched admission complete; logs archived to $ARCHIVE_DIR"
