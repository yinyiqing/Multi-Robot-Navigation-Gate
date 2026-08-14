#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$ROOT/TD3"
GATE_ROOT="$ROOT/experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究"
RUN_DIR="$GATE_ROOT/G11_F_epoch17_gate_v1/local_data/pilot"
LOG_DIR="$ROOT/logs/active/g11_f_epoch17_gate_pilot"
MANIFEST="$ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/g11_c_pilot_v1/validation.json.gz"
BASE_MODEL="TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best"
INTERACTION_MODEL="avoidance_actor_from_5a_balanced_continue_e20_s20260813_best"
DETECTOR="$ROOT/experiments/03_保留专门化/02_论文主线/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"
A1_GATE="$GATE_ROOT/G11_F_epoch17_gate_v1/local_data/a1_training/seed20260804/any/T1/best.pt"
B2_GATE="$GATE_ROOT/G11_F_epoch17_gate_v1/local_data/aggregated_training/seed20260804/any/T1/best.pt"
PID_FILE="$ROOT/.g11_f_epoch17_pilot.pid"
ROS_PORT=16623
GAZEBO_PORT=16723

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

exec 9>/tmp/local_critic_multi_robot_training.lock
flock -n 9 || { echo "Multi-robot evaluation lock is busy" >&2; exit 1; }
set +u
source /opt/ros/noetic/setup.bash
source "$ROOT/env.python.sh"
source "$ROOT/catkin_ws/devel_isolated/setup.bash"
set -u

export CUDA_VISIBLE_DEVICES=""
export ROS_HOSTNAME=localhost ROS_MASTER_URI="http://localhost:$ROS_PORT" ROS_PORT_SIM="$ROS_PORT"
export GAZEBO_MASTER_URI="http://localhost:$GAZEBO_PORT"
export GAZEBO_RESOURCE_PATH="$ROOT/catkin_ws/src/multi_robot_scenario/launch"
export DRL_MULTI_NUM_AGENTS=5
export DRL_MULTI_TEST_LAUNCHFILE=multi_robot_scenario_strong_interaction_pilot_5.launch
export DRL_MULTI_SCENARIO=manifest DRL_MULTI_MANIFEST_PATH="$MANIFEST" DRL_MULTI_MANIFEST_SAMPLING=cycle
export DRL_MULTI_TEST_TARGET_EPISODES=50 DRL_MULTI_STANDARD_ACTOR_FILE="$BASE_MODEL"
export DRL_MULTI_TEST_ACTOR_MODE=full DRL_MULTI_DENSE_ACTOR_MODE=full
export DRL_MULTI_FIXED_PHYSICS_STEP_SIZE=0.001 DRL_MULTI_REQUIRE_FIXED_STEP_SERVICE=1
unset DRL_MULTI_CASE_ORACLE_MAP DRL_MULTI_RULE_ORACLE_MODE
mkdir -p "$LOG_DIR" "$RUN_DIR/results" "$RUN_DIR/checkpoints"

verify_result() {
  /usr/bin/python3 - "$1" "$MANIFEST" <<'PY'
import gzip,json,sys,numpy as np
r=np.load(sys.argv[1],allow_pickle=True)
with gzip.open(sys.argv[2],'rt',encoding='utf-8') as f: ids=[str(x['scenario_id']) for x in json.load(f)['scenarios']]
if r.shape != (50,17) or [str(x[12]) for x in r] != ids: raise SystemExit(1)
if sum(int(x[6])+int(x[7])+int(x[10]) for x in r) != 250: raise SystemExit(1)
PY
}

run_one() {
  local policy="$1" repeat="$2" seed="$3" run="g11_f_c_${1}_r${2}_s${3}"
  local stats="$RUN_DIR/results/${run}.npy" state="$RUN_DIR/checkpoints/${run}_state.pt"
  local attempt log status
  if [[ -f "$stats" ]] && verify_result "$stats" 2>/dev/null; then echo "Skipping $run"; return; fi
  export DRL_MULTI_SEED="$seed" DRL_MULTI_TEST_FILE_NAME="$run"
  export DRL_MULTI_TEST_STATS_PATH="$stats" DRL_MULTI_TEST_STATE_PATH="$state"
  case "$policy" in
    5a)
      export DRL_MULTI_ACTOR_SELECTION_MODE=single
      unset DRL_MULTI_DENSE_ACTOR_FILE DRL_MULTI_GATE_DETECTOR_CHECKPOINT DRL_MULTI_GATE_CHECKPOINT
      unset DRL_MULTI_GATE_SWITCH_ON_THRESHOLD DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD
      unset DRL_MULTI_GATE_MINIMUM_HOLD_STEPS DRL_MULTI_GATE_EVALUATION_STRIDE ;;
    a1)
      export DRL_MULTI_ACTOR_SELECTION_MODE=learned_gate DRL_MULTI_DENSE_ACTOR_FILE="$INTERACTION_MODEL"
      export DRL_MULTI_GATE_DETECTOR_CHECKPOINT="$DETECTOR" DRL_MULTI_GATE_CHECKPOINT="$A1_GATE"
      export DRL_MULTI_GATE_SWITCH_ON_THRESHOLD=0.29 DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD=0.19
      export DRL_MULTI_GATE_MINIMUM_HOLD_STEPS=3 DRL_MULTI_GATE_EVALUATION_STRIDE=2 ;;
    b2)
      export DRL_MULTI_ACTOR_SELECTION_MODE=learned_gate DRL_MULTI_DENSE_ACTOR_FILE="$INTERACTION_MODEL"
      export DRL_MULTI_GATE_DETECTOR_CHECKPOINT="$DETECTOR" DRL_MULTI_GATE_CHECKPOINT="$B2_GATE"
      export DRL_MULTI_GATE_SWITCH_ON_THRESHOLD=0.43 DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD=0.33
      export DRL_MULTI_GATE_MINIMUM_HOLD_STEPS=3 DRL_MULTI_GATE_EVALUATION_STRIDE=2 ;;
  esac
  for attempt in 1 2 3; do
    log="$LOG_DIR/${run}_attempt${attempt}.log"
    echo "Starting $run attempt $attempt"
    set +e
    (cd "$TD3_DIR" && nice -n 10 python3 -u test_velodyne_td3_multi.py) >"$log" 2>&1
    status=$?
    set -e
    stop_runtime
    if [[ -f "$stats" ]] && verify_result "$stats" 2>/dev/null; then echo "Completed $run"; return; fi
    echo "$run attempt $attempt incomplete (exit=$status)"
  done
  echo "$run failed after 3 attempts" >&2; return 1
}

run_one 5a 1 20260805
run_one a1 1 20260805
run_one b2 1 20260805
run_one b2 2 20260806
run_one a1 2 20260806
run_one 5a 2 20260806
/usr/bin/python3 "$ROOT/scripts/analyze_g11_f_epoch17_pilot.py" >"$LOG_DIR/analysis.log" 2>&1
