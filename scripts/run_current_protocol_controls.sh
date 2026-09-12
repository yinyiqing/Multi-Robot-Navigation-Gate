#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
EXP="$BASE/35_当前协议对照补测"
MANIFEST="$BASE/34_双头RewardAwareGate/local_data/independent_test/manifest/dense_test_640_896.json.gz"
LOG_DIR="$EXP/logs"
RESULT_DIR="$EXP/local_data/results"
STATE_DIR="$EXP/local_data/checkpoints"
COMPLETION="$EXP/local_data/completion_n${EPISODES}.json"
LAUNCHFILE="$LOG_DIR/runtime_controls.launch"
PID_FILE="$ROOT/.current_protocol_controls.pid"
EPISODES="${CONTROL_TARGET_EPISODES:-16}"
SEEDS_TEXT="${CONTROL_SEEDS:-20260914}"
METHODS_TEXT="${CONTROL_METHODS:-always_on_interaction min_lidar_rule ttc_cpa_rule capacity_matched_actor proximity_only_router}"
ROS_PORT="${CONTROL_ROS_PORT:-18670}"
GAZEBO_PORT="${CONTROL_GAZEBO_PORT:-19670}"

FIVE_A="TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best"
EPOCH16="interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016"
CAPACITY="capacity_wide_r2b_5a_recipe_n5_seed20260823_best"
DETECTOR="$BASE/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"
B2="$BASE/11_可部署在线Gate研究/G11_B_student_rollout_v1/local_data/training/seed20260804/any/T1/best.pt"

[[ -f "$MANIFEST" ]] || { echo "missing manifest: $MANIFEST" >&2; exit 2; }
[[ -f "$DETECTOR" ]] || { echo "missing detector: $DETECTOR" >&2; exit 2; }
[[ -f "$B2" ]] || { echo "missing proximity checkpoint: $B2" >&2; exit 2; }

mkdir -p "$LOG_DIR" "$RESULT_DIR" "$STATE_DIR"
python3 "$ROOT/scripts/generate_multi_robot_launch.py" --num-agents 5 --output "$LAUNCHFILE"

stop_runtime() {
  local pgid children
  pgid="$(ps -o pgid= -p $$ | tr -d ' ')"
  children="$(ps -eo pid=,pgid= | awk -v p="$pgid" -v s="$$" '$2 == p && $1 != s {print $1}')"
  [[ -z "$children" ]] || xargs -r kill -TERM 2>/dev/null <<<"$children" || true
  sleep 2
  children="$(ps -eo pid=,pgid= | awk -v p="$pgid" -v s="$$" '$2 == p && $1 != s {print $1}')"
  [[ -z "$children" ]] || xargs -r kill -KILL 2>/dev/null <<<"$children" || true
  fuser -k -KILL "${ROS_PORT}/tcp" "${GAZEBO_PORT}/tcp" >/dev/null 2>&1 || true
}
cleanup() { stop_runtime; unlink "$PID_FILE" 2>/dev/null || true; }
trap cleanup EXIT

wait_for_ports() {
  for _ in $(seq 1 60); do
    if ! ss -ltnH | awk '{print $4}' | rg -q ":${ROS_PORT}$|:${GAZEBO_PORT}$"; then return 0; fi
    sleep 1
  done
  echo "ports did not become free" >&2
  return 1
}

verify_result() {
  python3 - "$1" "$MANIFEST" "$EPISODES" <<'PY'
import gzip,json,sys,numpy as np
rows=np.load(sys.argv[1],allow_pickle=True)
target=int(sys.argv[3])
if rows.shape != (target,17): raise SystemExit('wrong result shape')
with gzip.open(sys.argv[2],'rt',encoding='utf-8') as h:
    expected=[str(x['scenario_id']) for x in json.load(h)['scenarios'][:target]]
if [str(x) for x in rows[:,12]] != expected: raise SystemExit('scenario order mismatch')
if sum(int(r[6])+int(r[7])+int(r[10]) for r in rows) != target*5:
    raise SystemExit('terminal accounting mismatch')
PY
}

verify_partial() {
  python3 - "$1" "$2" "$MANIFEST" "$EPISODES" <<'PY'
import gzip,json,sys,numpy as np,torch
rows=np.load(sys.argv[1],allow_pickle=True)
state=torch.load(sys.argv[2],map_location='cpu',weights_only=False)
target=int(sys.argv[4])
if rows.ndim != 2 or rows.shape[1] != 17 or not 0 < len(rows) < target: raise SystemExit('invalid partial result')
if len(set(rows[:,12].tolist())) != len(rows): raise SystemExit('duplicate scenarios')
with gzip.open(sys.argv[3],'rt',encoding='utf-8') as h:
    expected=[str(x['scenario_id']) for x in json.load(h)['scenarios'][:len(rows)]]
if [str(x) for x in rows[:,12]] != expected: raise SystemExit('partial order mismatch')
if int(state.get('episode_num',-1)) not in (len(rows),len(rows)+1): raise SystemExit('state/result mismatch')
print(len(rows))
PY
}

clear_policy() {
  unset DRL_MULTI_STANDARD_ACTOR_FILE DRL_MULTI_DENSE_ACTOR_FILE DRL_MULTI_DENSE_ACTOR_MODE
  unset DRL_MULTI_ACTOR_SELECTION_MODE DRL_MULTI_GATE_DETECTOR_CHECKPOINT DRL_MULTI_GATE_CHECKPOINT
  unset DRL_MULTI_GATE_SWITCH_ON_THRESHOLD DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD
  unset DRL_MULTI_GATE_MINIMUM_HOLD_STEPS DRL_MULTI_GATE_EVALUATION_STRIDE
  unset DRL_MULTI_MIN_LIDAR_SWITCH_ON_DISTANCE DRL_MULTI_MIN_LIDAR_SWITCH_OFF_DISTANCE
  unset DRL_MULTI_MIN_LIDAR_MINIMUM_HOLD_STEPS DRL_MULTI_TTC_MINIMUM_HOLD_STEPS
  unset DRL_MULTI_TTC_EVALUATION_STRIDE
}

configure_policy() {
  local method="$1"
  clear_policy
  case "$method" in
    always_on_interaction)
      export DRL_MULTI_STANDARD_ACTOR_FILE="$EPOCH16" DRL_MULTI_ACTOR_SELECTION_MODE=single ;;
    capacity_matched_actor)
      export DRL_MULTI_STANDARD_ACTOR_FILE="$CAPACITY" DRL_MULTI_ACTOR_SELECTION_MODE=single ;;
    min_lidar_rule)
      export DRL_MULTI_STANDARD_ACTOR_FILE="$FIVE_A" DRL_MULTI_DENSE_ACTOR_FILE="$EPOCH16"
      export DRL_MULTI_DENSE_ACTOR_MODE=full DRL_MULTI_ACTOR_SELECTION_MODE=min_lidar_gate
      export DRL_MULTI_MIN_LIDAR_SWITCH_ON_DISTANCE=2.0 DRL_MULTI_MIN_LIDAR_SWITCH_OFF_DISTANCE=2.2
      export DRL_MULTI_MIN_LIDAR_MINIMUM_HOLD_STEPS=3 ;;
    ttc_cpa_rule)
      export DRL_MULTI_STANDARD_ACTOR_FILE="$FIVE_A" DRL_MULTI_DENSE_ACTOR_FILE="$EPOCH16"
      export DRL_MULTI_DENSE_ACTOR_MODE=full DRL_MULTI_ACTOR_SELECTION_MODE=ttc_cpa_gate
      export DRL_MULTI_GATE_DETECTOR_CHECKPOINT="$DETECTOR"
      export DRL_MULTI_TTC_MINIMUM_HOLD_STEPS=3 DRL_MULTI_TTC_EVALUATION_STRIDE=2 ;;
    proximity_only_router)
      export DRL_MULTI_STANDARD_ACTOR_FILE="$FIVE_A" DRL_MULTI_DENSE_ACTOR_FILE="$EPOCH16"
      export DRL_MULTI_DENSE_ACTOR_MODE=full DRL_MULTI_ACTOR_SELECTION_MODE=learned_gate
      export DRL_MULTI_GATE_DETECTOR_CHECKPOINT="$DETECTOR" DRL_MULTI_GATE_CHECKPOINT="$B2"
      export DRL_MULTI_GATE_SWITCH_ON_THRESHOLD=0.43 DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD=0.33
      export DRL_MULTI_GATE_MINIMUM_HOLD_STEPS=3 DRL_MULTI_GATE_EVALUATION_STRIDE=2 ;;
    *) echo "unknown method: $method" >&2; return 2 ;;
  esac
}

run_one() {
  local method="$1" seed="$2" run_name result state attempt status log progress
  run_name="controls_${method}_s${seed}_n${EPISODES}"
  result="$RESULT_DIR/${run_name}.npy"
  state="$STATE_DIR/${run_name}_state.pt"
  if [[ -f "$result" ]] && verify_result "$result" 2>/dev/null; then
    echo "Skipping completed $run_name"; return 0
  fi
  export DRL_MULTI_SEED="$seed" DRL_MULTI_TEST_FILE_NAME="$run_name"
  export DRL_MULTI_TEST_STATS_PATH="$result" DRL_MULTI_TEST_STATE_PATH="$state"
  configure_policy "$method"
  for attempt in $(seq 1 10); do
    log="$LOG_DIR/${run_name}_attempt${attempt}.log"
    echo "Starting $run_name attempt $attempt"
    wait_for_ports
    set +e
    (cd "$ROOT/TD3" && nice -n 10 python3 -u test_velodyne_td3_multi.py) >"$log" 2>&1
    status=$?
    set -e
    stop_runtime
    wait_for_ports
    if [[ -f "$result" ]] && verify_result "$result" 2>/dev/null; then
      echo "Completed $run_name"; return 0
    fi
    if [[ ! -f "$result" || ! -f "$state" ]]; then
      echo "$run_name failed before resumable state (exit=$status)" >&2; return 1
    fi
    progress="$(verify_partial "$result" "$state")" || return 1
    echo "$run_name interrupted at $progress/$EPISODES; exact resume"
  done
  echo "$run_name failed after retries" >&2; return 1
}

exec 9>/tmp/local_critic_multi_robot_training.lock
flock 9
set +u
source /opt/ros/noetic/setup.bash
source "$ROOT/env.python.sh"
source "$ROOT/catkin_ws/devel_isolated/setup.bash"
set -u
export CUDA_VISIBLE_DEVICES="" ROS_HOSTNAME=localhost ROS_MASTER_URI="http://localhost:$ROS_PORT" ROS_PORT_SIM="$ROS_PORT"
export GAZEBO_MASTER_URI="http://localhost:$GAZEBO_PORT" GAZEBO_IP=127.0.0.1
export GAZEBO_RESOURCE_PATH="$ROOT/catkin_ws/src/multi_robot_scenario/launch"
export DRL_MULTI_NUM_AGENTS=5 DRL_MULTI_TEST_LAUNCHFILE="$LAUNCHFILE"
export DRL_MULTI_SCENARIO=manifest DRL_MULTI_MANIFEST_PATH="$MANIFEST" DRL_MULTI_MANIFEST_SAMPLING=cycle
export DRL_MULTI_TEST_TARGET_EPISODES="$EPISODES" DRL_MULTI_FIXED_PHYSICS_STEP_SIZE=0.001
export DRL_MULTI_REQUIRE_FIXED_STEP_SERVICE=1 DRL_MULTI_TEST_ACTOR_MODE=full DRL_MULTI_DENSE_ACTOR_MODE=full
export DRL_MULTI_TEST_DISABLE_TENSORBOARD=1

read -r -a SEEDS <<<"$SEEDS_TEXT"
read -r -a METHODS <<<"$METHODS_TEXT"
for seed in "${SEEDS[@]}"; do
  for method in "${METHODS[@]}"; do
    run_one "$method" "$seed"
  done
done

python3 - "$RESULT_DIR" "$SEEDS_TEXT" "$METHODS_TEXT" "$EPISODES" "$MANIFEST" "$COMPLETION" <<'PY'
import hashlib,json,sys
from pathlib import Path
result_dir,seeds_text,methods_text,episodes,manifest,out=sys.argv[1:]
seeds=[int(x) for x in seeds_text.split()]; methods=methods_text.split(); episodes=int(episodes)
result_dir=Path(result_dir); hashes={}; missing=[]
for method in methods:
  for seed in seeds:
    p=result_dir/f'controls_{method}_s{seed}_n{episodes}.npy'
    if p.is_file(): hashes[f'{method}_s{seed}']=hashlib.sha256(p.read_bytes()).hexdigest()
    else: missing.append(str(p))
record={'format_version':1,'status':'complete' if not missing else 'incomplete','experiment_id':'current-protocol-controls','manifest':str(Path(manifest)),'manifest_sha256':hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),'episodes_per_method_repeat':episodes,'seeds':seeds,'methods':methods,'total_episodes':episodes*len(seeds)*len(methods),'result_sha256':hashes,'missing':missing,'actor_or_router_updated':False,'performance_early_stop':False}
Path(out).write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
PY
echo "Current-protocol controls complete"
