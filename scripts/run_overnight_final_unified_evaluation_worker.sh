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
N5="current_generalist_n5_original_broad_s20260810_best"
DETECTOR="$BASE/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"
F_A1="$BASE/11_可部署在线Gate研究/G11_F_epoch17_gate_v1/local_data/a1_training/seed20260804/any/T1/best.pt"
F_B2="$BASE/11_可部署在线Gate研究/G11_F_epoch17_gate_v1/local_data/aggregated_training/seed20260804/any/T1/best.pt"

verify_sha() {
  local path="$1" expected="$2"
  [[ -f "$path" ]] || { echo "Missing frozen input: $path" >&2; exit 1; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]] || {
    echo "SHA-256 mismatch: $path" >&2
    exit 1
  }
}

verify_sha "$MANIFEST" "52435d6c5bdf9914e7212dd29cb4bfec074257f72d85f0d71741deee7c63b635"
verify_sha "$ROOT/TD3/pytorch_models/${FIVE_A}_actor.pth" "fa28855049b67b3ee44c66d55d4f14441fc7c521e5429862c75b152f7d5cacc5"
verify_sha "$ROOT/TD3/pytorch_models/${EPOCH17}_actor.pth" "149c2e42848ecc9bc478cbed7fd89b9062936dbd5c669b55e6964441685155a5"
verify_sha "$ROOT/TD3/pytorch_models/${R2}_actor.pth" "ace910553931873a275d66e3a964fd2b4716d30b6c68c8dcb3e7af96e56783ee"
verify_sha "$ROOT/TD3/pytorch_models/${N5}_actor.pth" "53964e12c2d6c5f0855530f22bdd721170b911640883c7616b14dc21aa12cfeb"
verify_sha "$DETECTOR" "0b914c0d090bbaba0a2be63c0d75d88580bf4d71778a7f1749c63989e3dbbd56"
verify_sha "$F_A1" "b28e81d341c145d6fa8c881dd98c7ece5285231e7d080b3f71afcd2dfe3a0beb"

pgrep -af '^python3(\.8)? -u train_velodyne_td3_multi.py($| )' >/dev/null && {
  echo "Another multi-robot Actor run is active" >&2
  exit 1
}

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
  unset DRL_MULTI_RECOVERY_ORACLE_CANDIDATE_DISTANCE DRL_MULTI_RECOVERY_ORACLE_RELEASE_DISTANCE
  unset DRL_MULTI_RECOVERY_ORACLE_PROGRESS_THRESHOLD DRL_MULTI_RECOVERY_ORACLE_PROGRESS_WINDOW
  unset DRL_MULTI_RECOVERY_ORACLE_DISTANCE_DELTA_THRESHOLD DRL_MULTI_RECOVERY_ORACLE_GOAL_DISTANCE
  unset DRL_MULTI_RECOVERY_ORACLE_MINIMUM_HOLD_STEPS DRL_MULTI_RECOVERY_ORACLE_MAXIMUM_HOLD_STEPS
  case "$policy" in
    5a) export DRL_MULTI_STANDARD_ACTOR_FILE="$FIVE_A" DRL_MULTI_ACTOR_SELECTION_MODE=single ;;
    epoch17) export DRL_MULTI_STANDARD_ACTOR_FILE="$EPOCH17" DRL_MULTI_ACTOR_SELECTION_MODE=single ;;
    r2) export DRL_MULTI_STANDARD_ACTOR_FILE="$R2" DRL_MULTI_ACTOR_SELECTION_MODE=single ;;
    n5) export DRL_MULTI_STANDARD_ACTOR_FILE="$N5" DRL_MULTI_ACTOR_SELECTION_MODE=single ;;
    f_a1) export DRL_MULTI_STANDARD_ACTOR_FILE="$FIVE_A" DRL_MULTI_DENSE_ACTOR_FILE="$EPOCH17"
      export DRL_MULTI_ACTOR_SELECTION_MODE=learned_gate DRL_MULTI_GATE_DETECTOR_CHECKPOINT="$DETECTOR" DRL_MULTI_GATE_CHECKPOINT="$F_A1"
      export DRL_MULTI_GATE_SWITCH_ON_THRESHOLD=0.29 DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD=0.19 DRL_MULTI_GATE_MINIMUM_HOLD_STEPS=3 DRL_MULTI_GATE_EVALUATION_STRIDE=2 ;;
    f_b2) export DRL_MULTI_STANDARD_ACTOR_FILE="$FIVE_A" DRL_MULTI_DENSE_ACTOR_FILE="$EPOCH17"
      export DRL_MULTI_ACTOR_SELECTION_MODE=learned_gate DRL_MULTI_GATE_DETECTOR_CHECKPOINT="$DETECTOR" DRL_MULTI_GATE_CHECKPOINT="$F_B2"
      export DRL_MULTI_GATE_SWITCH_ON_THRESHOLD=0.43 DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD=0.33 DRL_MULTI_GATE_MINIMUM_HOLD_STEPS=3 DRL_MULTI_GATE_EVALUATION_STRIDE=2 ;;
    n5_recovery) export DRL_MULTI_STANDARD_ACTOR_FILE="$N5" DRL_MULTI_DENSE_ACTOR_FILE="$EPOCH17"
      export DRL_MULTI_ACTOR_SELECTION_MODE=recovery_oracle DRL_MULTI_DENSE_ACTOR_MODE=full
      export DRL_MULTI_RECOVERY_ORACLE_CANDIDATE_DISTANCE=2.0 DRL_MULTI_RECOVERY_ORACLE_RELEASE_DISTANCE=2.4
      export DRL_MULTI_RECOVERY_ORACLE_PROGRESS_THRESHOLD=0.003 DRL_MULTI_RECOVERY_ORACLE_PROGRESS_WINDOW=5
      export DRL_MULTI_RECOVERY_ORACLE_DISTANCE_DELTA_THRESHOLD=0.02 DRL_MULTI_RECOVERY_ORACLE_GOAL_DISTANCE=0.45
      export DRL_MULTI_RECOVERY_ORACLE_MINIMUM_HOLD_STEPS=3 DRL_MULTI_RECOVERY_ORACLE_MAXIMUM_HOLD_STEPS=20 ;;
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

run_one n5 20260830
run_one r2 20260830
run_one f_a1 20260830
run_one n5_recovery 20260830

run_one n5_recovery 20260831
run_one f_a1 20260831
run_one r2 20260831
run_one n5 20260831

set +e
for candidate in r2 f_a1 n5_recovery; do
  /usr/bin/python3 "$ROOT/scripts/compare_actor_validation.py" \
    "$RESULT_DIR/g20_n5_s20260830.npy" "$RESULT_DIR/g20_${candidate}_s20260830.npy" \
    --baseline-label N5 --candidate-label "$candidate" --manifest "$MANIFEST" \
    --output "$RESULT_DIR/compare_n5_${candidate}_s20260830.json" \
    >>"$LOG_DIR/analysis.log" 2>&1
  /usr/bin/python3 "$ROOT/scripts/compare_actor_validation.py" \
    "$RESULT_DIR/g20_n5_s20260831.npy" "$RESULT_DIR/g20_${candidate}_s20260831.npy" \
    --baseline-label N5 --candidate-label "$candidate" --manifest "$MANIFEST" \
    --output "$RESULT_DIR/compare_n5_${candidate}_s20260831.json" \
    >>"$LOG_DIR/analysis.log" 2>&1
done
set -e
/usr/bin/python3 "$ROOT/scripts/analyze_g20_mainline_adjudication.py" \
  >"$LOG_DIR/combined_analysis.log" 2>&1
[[ ! -e "$ARCHIVE_DIR" ]] || { echo "Archive exists: $ARCHIVE_DIR" >&2; exit 1; }
mkdir -p "$(dirname "$ARCHIVE_DIR")"
mv "$LOG_DIR" "$ARCHIVE_DIR"
