#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$ROOT/TD3"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
RUN_DIR="$BASE/20_夜间最终统一评测/local_data"
RESULT_DIR="$RUN_DIR/results"
STATE_DIR="$RUN_DIR/checkpoints"
LOG_DIR="$ROOT/logs/active/g20-overnight-final-unified"
ARCHIVE_DIR="$ROOT/logs/archive/validation/g20_overnight_final_unified"
MANIFEST="$BASE/datasets/fixed_v1/views/g12_full_scene_selection_v1/validation.json.gz"
LAUNCHFILE="$LOG_DIR/runtime_g20_overnight_final_unified.launch"
PID_FILE="$ROOT/.g20_overnight_final_unified.pid"
LOCK_FILE=/tmp/local_critic_multi_robot_training.lock
ROS_PORT=17623
GAZEBO_PORT=17723

FIVE_A="TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best"
EPOCH17="avoidance_actor_from_5a_balanced_continue_e20_s20260813_best"
R2="capacity_wide_r2_s4_broad_n5_seed20260816_epoch_001"
DETECTOR="$BASE/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"
F_A1="$BASE/11_可部署在线Gate研究/G11_F_epoch17_gate_v1/local_data/a1_training/seed20260804/any/T1/best.pt"
F_B2="$BASE/11_可部署在线Gate研究/G11_F_epoch17_gate_v1/local_data/aggregated_training/seed20260804/any/T1/best.pt"

[[ -f "$MANIFEST" && -f "$ROOT/TD3/pytorch_models/${FIVE_A}_actor.pth" && \
   -f "$ROOT/TD3/pytorch_models/${EPOCH17}_actor.pth" && \
   -f "$ROOT/TD3/pytorch_models/${R2}_actor.pth" && \
   -f "$DETECTOR" && -f "$F_A1" && -f "$F_B2" ]] || {
  echo "Required frozen evaluation artifact is missing" >&2
  exit 1
}

while [[ -f "$ROOT/.g12_r2c_corrected.pid" ]] || \
      pgrep -af 'python3 -u train_velodyne_td3_multi.py' >/dev/null; do
  echo "Waiting for R2C training to finish..."
  sleep 30
done

mkdir -p "$RESULT_DIR" "$STATE_DIR"
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

exec 9>"$LOCK_FILE"
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
export DRL_MULTI_TEST_TARGET_EPISODES=120 DRL_MULTI_FIXED_PHYSICS_STEP_SIZE=0.001
export DRL_MULTI_REQUIRE_FIXED_STEP_SERVICE=1 DRL_MULTI_TEST_ACTOR_MODE=full

verify_result() {
  /usr/bin/python3 - "$1" "$MANIFEST" <<'PY'
import gzip, json, sys, numpy as np
r = np.load(sys.argv[1], allow_pickle=True)
with gzip.open(sys.argv[2], 'rt', encoding='utf-8') as f:
    ids = [str(x['scenario_id']) for x in json.load(f)['scenarios'][:120]]
if r.shape != (120, 17) or [str(x[12]) for x in r] != ids:
    raise SystemExit(1)
if sum(int(x[6]) + int(x[7]) + int(x[10]) for x in r) != 600:
    raise SystemExit(1)
PY
}

run_one() {
  local policy="$1"
  local seed="$2"
  local run="g20_${policy}_s${seed}"
  local result="$RESULT_DIR/${run}.npy" state="$STATE_DIR/${run}_state.pt"
  local attempt log status
  if [[ -f "$result" ]] && verify_result "$result" 2>/dev/null; then
    echo "Skipping completed $run"
    return
  fi
  export DRL_MULTI_SEED="$seed" DRL_MULTI_TEST_FILE_NAME="$run"
  export DRL_MULTI_TEST_STATS_PATH="$result" DRL_MULTI_TEST_STATE_PATH="$state"
  unset DRL_MULTI_DENSE_ACTOR_FILE DRL_MULTI_GATE_DETECTOR_CHECKPOINT DRL_MULTI_GATE_CHECKPOINT
  unset DRL_MULTI_GATE_SWITCH_ON_THRESHOLD DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD
  unset DRL_MULTI_GATE_MINIMUM_HOLD_STEPS DRL_MULTI_GATE_EVALUATION_STRIDE
  case "$policy" in
    5a) export DRL_MULTI_STANDARD_ACTOR_FILE="$FIVE_A" DRL_MULTI_ACTOR_SELECTION_MODE=single ;;
    epoch17) export DRL_MULTI_STANDARD_ACTOR_FILE="$EPOCH17" DRL_MULTI_ACTOR_SELECTION_MODE=single ;;
    r2) export DRL_MULTI_STANDARD_ACTOR_FILE="$R2" DRL_MULTI_ACTOR_SELECTION_MODE=single ;;
    f_a1) export DRL_MULTI_STANDARD_ACTOR_FILE="$FIVE_A" DRL_MULTI_DENSE_ACTOR_FILE="$EPOCH17"
      export DRL_MULTI_ACTOR_SELECTION_MODE=learned_gate DRL_MULTI_GATE_DETECTOR_CHECKPOINT="$DETECTOR" DRL_MULTI_GATE_CHECKPOINT="$F_A1"
      export DRL_MULTI_GATE_SWITCH_ON_THRESHOLD=0.29 DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD=0.19 DRL_MULTI_GATE_MINIMUM_HOLD_STEPS=3 DRL_MULTI_GATE_EVALUATION_STRIDE=2 ;;
    f_b2) export DRL_MULTI_STANDARD_ACTOR_FILE="$FIVE_A" DRL_MULTI_DENSE_ACTOR_FILE="$EPOCH17"
      export DRL_MULTI_ACTOR_SELECTION_MODE=learned_gate DRL_MULTI_GATE_DETECTOR_CHECKPOINT="$DETECTOR" DRL_MULTI_GATE_CHECKPOINT="$F_B2"
      export DRL_MULTI_GATE_SWITCH_ON_THRESHOLD=0.43 DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD=0.33 DRL_MULTI_GATE_MINIMUM_HOLD_STEPS=3 DRL_MULTI_GATE_EVALUATION_STRIDE=2 ;;
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
    echo "$run attempt $attempt incomplete (exit=$status)"
  done
  echo "$run failed after 3 attempts" >&2
  return 1
}

for seed in 20260830 20260831; do
  run_one 5a "$seed"
  run_one epoch17 "$seed"
  run_one r2 "$seed"
  run_one f_a1 "$seed"
  run_one f_b2 "$seed"
done

set +e
for candidate in epoch17 r2 f_a1 f_b2; do
  /usr/bin/python3 "$ROOT/scripts/compare_actor_validation.py" \
    "$RESULT_DIR/g20_5a_s20260830.npy" "$RESULT_DIR/g20_${candidate}_s20260830.npy" \
    --baseline-label 5A --candidate-label "$candidate" --manifest "$MANIFEST" \
    --output "$RESULT_DIR/compare_5a_${candidate}_s20260830.json" \
    >>"$LOG_DIR/analysis.log" 2>&1
  /usr/bin/python3 "$ROOT/scripts/compare_actor_validation.py" \
    "$RESULT_DIR/g20_5a_s20260831.npy" "$RESULT_DIR/g20_${candidate}_s20260831.npy" \
    --baseline-label 5A --candidate-label "$candidate" --manifest "$MANIFEST" \
    --output "$RESULT_DIR/compare_5a_${candidate}_s20260831.json" \
    >>"$LOG_DIR/analysis.log" 2>&1
done
set -e
[[ ! -e "$ARCHIVE_DIR" ]] || { echo "Archive exists: $ARCHIVE_DIR" >&2; exit 1; }
mkdir -p "$(dirname "$ARCHIVE_DIR")"
mv "$LOG_DIR" "$ARCHIVE_DIR"
