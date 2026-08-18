#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
RUN_DIR="$BASE/25_最终消融与Sealed评测/local_data"
RESULT_DIR="$RUN_DIR/results"
STATE_DIR="$RUN_DIR/checkpoints"
MANIFEST="$BASE/datasets/fixed_v1/dense/validation.json.gz"
LOG_DIR="$ROOT/logs/active/g25-router-validation"
ARCHIVE_DIR="$ROOT/logs/archive/validation/g25_router_ablations"
PID_FILE="$ROOT/.g25_router_validation.pid"
LAUNCHFILE="$LOG_DIR/runtime_g25_router_validation.launch"
FIVE_A="TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best"
EPOCH16="interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016"
DETECTOR="$BASE/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"
V4="$RUN_DIR/router_ablations/v4_single_frame/any/T1/best.pt"
V5="$RUN_DIR/router_ablations/v5_no_action_difference/any/T1/best.pt"
ROS_PORT=17823
GAZEBO_PORT=17923
EPISODES=256
SEED=20260810

stop_runtime_children() {
  local pgid child_pids
  pgid="$(ps -o pgid= -p $$ | tr -d ' ')"
  child_pids="$(ps -eo pid=,pgid= | awk -v p="$pgid" -v s="$$" '$2 == p && $1 != s {print $1}')"
  [[ -z "$child_pids" ]] || xargs -r kill -TERM 2>/dev/null <<<"$child_pids" || true
  sleep 3
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
export DRL_MULTI_REQUIRE_FIXED_STEP_SERVICE=1 DRL_MULTI_TEST_ACTOR_MODE=full
export DRL_MULTI_DENSE_ACTOR_MODE=full DRL_MULTI_SEED="$SEED"
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
import gzip,json,sys,numpy as np
rows=np.load(sys.argv[1],allow_pickle=True)
episodes=int(sys.argv[3])
with gzip.open(sys.argv[2],'rt',encoding='utf-8') as f:
    expected=[str(x['scenario_id']) for x in json.load(f)['scenarios'][:episodes]]
if rows.shape != (episodes,17): raise SystemExit('wrong result shape')
if [str(x) for x in rows[:,12]] != expected: raise SystemExit('scenario order mismatch')
if sum(int(x[6])+int(x[7])+int(x[10]) for x in rows) != episodes*5:
    raise SystemExit('terminal accounting mismatch')
PY
}

verify_partial_result() {
  python3 - "$1" "$2" "$EPISODES" <<'PY'
import sys,numpy as np,torch
rows=np.load(sys.argv[1],allow_pickle=True)
state=torch.load(sys.argv[2],map_location='cpu',weights_only=False)
target=int(sys.argv[3])
if rows.ndim != 2 or rows.shape[1] != 17 or not 0 < len(rows) < target:
    raise SystemExit('invalid partial result')
if len(set(rows[:,12].tolist())) != len(rows): raise SystemExit('duplicate scenarios')
if int(state.get('episode_num',-1)) != len(rows): raise SystemExit('state/result mismatch')
manifest_state=state.get('manifest_sampling_state') or {}
if int(manifest_state.get('curriculum_case_index',-1)) != len(rows):
    raise SystemExit('manifest state mismatch')
print(len(rows))
PY
}

configure_policy() {
  local policy="$1"
  export DRL_MULTI_STANDARD_ACTOR_FILE="$FIVE_A" DRL_MULTI_DENSE_ACTOR_FILE="$EPOCH16"
  export DRL_MULTI_ACTOR_SELECTION_MODE=learned_gate
  export DRL_MULTI_GATE_DETECTOR_CHECKPOINT="$DETECTOR"
  export DRL_MULTI_GATE_MINIMUM_HOLD_STEPS=3 DRL_MULTI_GATE_EVALUATION_STRIDE=2
  case "$policy" in
    v4_single_frame)
      export DRL_MULTI_GATE_CHECKPOINT="$V4"
      export DRL_MULTI_GATE_SWITCH_ON_THRESHOLD=0.43 DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD=0.33 ;;
    v5_no_action_difference)
      export DRL_MULTI_GATE_CHECKPOINT="$V5"
      export DRL_MULTI_GATE_SWITCH_ON_THRESHOLD=0.51 DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD=0.41 ;;
    *) echo "Unknown G25 policy: $policy" >&2; return 2 ;;
  esac
}

run_one() {
  local policy="$1" run_name="g25_dense256_${1}_s${SEED}"
  local result="$RESULT_DIR/${run_name}.npy" state="$STATE_DIR/${run_name}_state.pt"
  local attempt status log progress
  if [[ -f "$result" ]] && verify_result "$result" 2>/dev/null; then
    echo "Skipping completed $run_name"
    return 0
  fi
  export DRL_MULTI_TEST_FILE_NAME="$run_name" DRL_MULTI_TEST_STATS_PATH="$result"
  export DRL_MULTI_TEST_STATE_PATH="$state"
  configure_policy "$policy"
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

run_one v4_single_frame
run_one v5_no_action_difference

[[ ! -e "$ARCHIVE_DIR" ]] || { echo "Archive exists: $ARCHIVE_DIR" >&2; exit 1; }
mkdir -p "$(dirname "$ARCHIVE_DIR")"
mv "$LOG_DIR" "$ARCHIVE_DIR"
echo "G25 Router validation complete: $ARCHIVE_DIR"
