#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
G27="$BASE/27_反事实Reward增强Gate监督"
MANIFEST="$BASE/datasets/fixed_v1/dense/validation.json.gz"
CHECKPOINT="$G27/local_data/training/seed20260910/best.pt"
RESULT_DIR="$G27/local_data/validation/results"
STATE_DIR="$G27/local_data/validation/checkpoints"
RESULT="$RESULT_DIR/g27_dense256_b3_s20260810.npy"
STATE="$STATE_DIR/g27_dense256_b3_s20260810_state.pt"
LOG_DIR="$G27/logs/p3_validation"
PID_FILE="$ROOT/.g27_p3_validation.pid"
LAUNCHFILE="$LOG_DIR/runtime_g27_p3.launch"
ROS_PORT=18510
GAZEBO_PORT=19510
FIVE_A="TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best"
EPOCH16="interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016"
DETECTOR="$BASE/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"

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

cleanup() {
  stop_runtime
  unlink "$PID_FILE" 2>/dev/null || true
}
trap cleanup EXIT

verify_result() {
  python3 - "$RESULT" "$MANIFEST" <<'PY'
import gzip,json,sys,numpy as np
rows=np.load(sys.argv[1],allow_pickle=True)
with gzip.open(sys.argv[2],'rt',encoding='utf-8') as f:
    expected=[str(x['scenario_id']) for x in json.load(f)['scenarios'][:256]]
if rows.shape != (256,17): raise SystemExit('wrong result shape')
if [str(x) for x in rows[:,12]] != expected: raise SystemExit('scenario order mismatch')
if sum(int(x[6])+int(x[7])+int(x[10]) for x in rows) != 1280:
    raise SystemExit('terminal accounting mismatch')
PY
}

verify_partial() {
  python3 - "$RESULT" "$STATE" <<'PY'
import sys,numpy as np,torch
rows=np.load(sys.argv[1],allow_pickle=True)
state=torch.load(sys.argv[2],map_location='cpu',weights_only=False)
if rows.ndim != 2 or rows.shape[1] != 17 or not 0 < len(rows) < 256:
    raise SystemExit('invalid partial result')
if len(set(rows[:,12].tolist())) != len(rows): raise SystemExit('duplicate scenarios')
if int(state.get('episode_num',-1)) != len(rows): raise SystemExit('state/result mismatch')
manifest_state=state.get('manifest_sampling_state') or {}
if int(manifest_state.get('curriculum_case_index',-1)) != len(rows):
    raise SystemExit('manifest state mismatch')
print(len(rows))
PY
}

mkdir -p "$RESULT_DIR" "$STATE_DIR" "$LOG_DIR"
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
export DRL_MULTI_REQUIRE_FIXED_STEP_SERVICE=1 DRL_MULTI_TEST_ACTOR_MODE=full
export DRL_MULTI_DENSE_ACTOR_MODE=full DRL_MULTI_SEED=20260810
export DRL_MULTI_TEST_FILE_NAME=g27_dense256_b3_s20260810
export DRL_MULTI_TEST_STATS_PATH="$RESULT" DRL_MULTI_TEST_STATE_PATH="$STATE"
export DRL_MULTI_STANDARD_ACTOR_FILE="$FIVE_A" DRL_MULTI_DENSE_ACTOR_FILE="$EPOCH16"
export DRL_MULTI_ACTOR_SELECTION_MODE=learned_gate
export DRL_MULTI_GATE_DETECTOR_CHECKPOINT="$DETECTOR" DRL_MULTI_GATE_CHECKPOINT="$CHECKPOINT"
export DRL_MULTI_GATE_SWITCH_ON_THRESHOLD=0.43 DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD=0.33
export DRL_MULTI_GATE_MINIMUM_HOLD_STEPS=3 DRL_MULTI_GATE_EVALUATION_STRIDE=2

if [[ -f "$RESULT" ]] && verify_result 2>/dev/null; then
  echo "B3 validation already complete; reusing it"
else
  for attempt in $(seq 1 10); do
    echo "Starting B3 validation attempt $attempt"
    set +e
    (cd "$ROOT/TD3" && python3 -u test_velodyne_td3_multi.py) \
      >"$LOG_DIR/b3_attempt${attempt}.log" 2>&1
    status=$?
    set -e
    stop_runtime
    if [[ -f "$RESULT" ]] && verify_result 2>/dev/null; then
      echo "B3 validation completed"
      break
    fi
    if [[ ! -f "$RESULT" || ! -f "$STATE" ]]; then
      echo "B3 validation failed before resumable state (exit=$status)" >&2
      exit 1
    fi
    progress="$(verify_partial)" || exit 1
    echo "B3 validation interrupted at $progress/256; resuming exact run"
    if [[ "$attempt" == 10 ]]; then
      echo "B3 validation failed after 10 infrastructure attempts" >&2
      exit 1
    fi
  done
fi

python3 "$ROOT/scripts/analyze_g27_p3_validation.py"
