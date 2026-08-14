#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$ROOT/TD3"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
RUN_DIR="$BASE/17_完整场景统一对比/local_data"
LOG_DIR="$ROOT/logs/active/g17-full-scene-comparison"
MANIFEST="$BASE/datasets/fixed_v1/views/g12_full_scene_selection_v1/validation.json.gz"
LAUNCHFILE="$LOG_DIR/runtime_g17_full_scene.launch"
FIVE_A="TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best"
R2B="capacity_wide_r2b_5a_recipe_n5_seed20260823_best"
INTERACTION="avoidance_actor_from_5a_balanced_continue_e20_s20260813_best"
DETECTOR="$BASE/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"
GATE="$BASE/11_可部署在线Gate研究/G11_F_epoch17_gate_v1/local_data/a1_training/seed20260804/any/T1/best.pt"

POLICY="${1:?policy required}"; SEED="${2:?seed required}"
ROS_PORT="${3:?ROS port required}"; GAZEBO_PORT="${4:?Gazebo port required}"
RUN="g17_${POLICY}_s${SEED}"
RESULT_DIR="$RUN_DIR/results"; STATE_DIR="$RUN_DIR/checkpoints"
STATS="$RESULT_DIR/${RUN}.npy"; STATE="$STATE_DIR/${RUN}_state.pt"

stop_runtime() {
  fuser -k -TERM "${ROS_PORT}/tcp" "${GAZEBO_PORT}/tcp" >/dev/null 2>&1 || true
  sleep 2
  fuser -k -KILL "${ROS_PORT}/tcp" "${GAZEBO_PORT}/tcp" >/dev/null 2>&1 || true
}
trap stop_runtime EXIT

exec 9>"/tmp/local_critic_multi_robot_training_${ROS_PORT}.lock"
flock -n 9 || { echo "Port-specific evaluation lock is busy: $ROS_PORT" >&2; exit 1; }
set +u
source /opt/ros/noetic/setup.bash
source "$ROOT/env.python.sh"
source "$ROOT/catkin_ws/devel_isolated/setup.bash"
set -u

export CUDA_VISIBLE_DEVICES=""
export ROS_HOSTNAME=localhost ROS_MASTER_URI="http://localhost:$ROS_PORT" ROS_PORT_SIM="$ROS_PORT"
export GAZEBO_MASTER_URI="http://localhost:$GAZEBO_PORT" GAZEBO_RESOURCE_PATH="$ROOT/catkin_ws/src/multi_robot_scenario/launch"
export DRL_MULTI_NUM_AGENTS=5 DRL_MULTI_TEST_LAUNCHFILE="$LAUNCHFILE"
export DRL_MULTI_SCENARIO=manifest DRL_MULTI_MANIFEST_PATH="$MANIFEST" DRL_MULTI_MANIFEST_SAMPLING=cycle
export DRL_MULTI_TEST_TARGET_EPISODES=120 DRL_MULTI_FIXED_PHYSICS_STEP_SIZE=0.001
export DRL_MULTI_REQUIRE_FIXED_STEP_SERVICE=1 DRL_MULTI_TEST_ACTOR_MODE=full
mkdir -p "$LOG_DIR" "$RESULT_DIR" "$STATE_DIR"

case "$POLICY" in
  5a|r2bbest)
    export DRL_MULTI_ACTOR_SELECTION_MODE=single
    if [[ "$POLICY" == 5a ]]; then export DRL_MULTI_STANDARD_ACTOR_FILE="$FIVE_A"; else export DRL_MULTI_STANDARD_ACTOR_FILE="$R2B"; fi
    unset DRL_MULTI_DENSE_ACTOR_FILE DRL_MULTI_GATE_DETECTOR_CHECKPOINT DRL_MULTI_GATE_CHECKPOINT
    unset DRL_MULTI_GATE_SWITCH_ON_THRESHOLD DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD DRL_MULTI_GATE_MINIMUM_HOLD_STEPS DRL_MULTI_GATE_EVALUATION_STRIDE ;;
  a1)
    export DRL_MULTI_STANDARD_ACTOR_FILE="$FIVE_A" DRL_MULTI_DENSE_ACTOR_FILE="$INTERACTION"
    export DRL_MULTI_DENSE_ACTOR_MODE=full DRL_MULTI_ACTOR_SELECTION_MODE=learned_gate
    export DRL_MULTI_GATE_DETECTOR_CHECKPOINT="$DETECTOR" DRL_MULTI_GATE_CHECKPOINT="$GATE"
    export DRL_MULTI_GATE_SWITCH_ON_THRESHOLD=0.29 DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD=0.19
    export DRL_MULTI_GATE_MINIMUM_HOLD_STEPS=3 DRL_MULTI_GATE_EVALUATION_STRIDE=2 ;;
  *) echo "Unknown policy: $POLICY" >&2; exit 2 ;;
esac

verify_result() {
  /usr/bin/python3 - "$1" "$MANIFEST" <<'PY'
import gzip,json,sys,numpy as np
r=np.load(sys.argv[1],allow_pickle=True)
with gzip.open(sys.argv[2],'rt',encoding='utf-8') as f: ids=[str(x['scenario_id']) for x in json.load(f)['scenarios']]
if r.shape != (120,17) or [str(x[12]) for x in r] != ids or len(set(ids)) != 120: raise SystemExit(1)
if sum(int(x[6])+int(x[7])+int(x[10]) for x in r) != 600: raise SystemExit(1)
PY
}

if [[ -f "$STATS" ]] && verify_result "$STATS" 2>/dev/null; then echo "Skipping completed $RUN"; exit 0; fi
for attempt in 1 2 3; do
  log="$LOG_DIR/${RUN}_attempt${attempt}.log"
  echo "Starting $RUN attempt $attempt policy=$POLICY port=$ROS_PORT"
  set +e; (cd "$TD3_DIR" && nice -n 10 python3 -u test_velodyne_td3_multi.py) >"$log" 2>&1; status=$?; set -e
  stop_runtime
  if [[ -f "$STATS" ]] && verify_result "$STATS" 2>/dev/null; then echo "Completed $RUN"; exit 0; fi
  echo "$RUN attempt $attempt incomplete (exit=$status)"
done
echo "$RUN failed after 3 attempts" >&2; exit 1
