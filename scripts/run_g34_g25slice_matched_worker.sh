#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
RUN="$BASE/36_G34_G25slice_matched"
MANIFEST="${G34_MATCHED_MANIFEST:?G34_MATCHED_MANIFEST is required}"
LOG_DIR="${G34_MATCHED_LOG_DIR:?G34_MATCHED_LOG_DIR is required}"
RESULT_DIR="${G34_MATCHED_RESULT_DIR:?G34_MATCHED_RESULT_DIR is required}"
STATE_DIR="${G34_MATCHED_STATE_DIR:?G34_MATCHED_STATE_DIR is required}"
PID_FILE="${G34_MATCHED_PID_FILE:?G34_MATCHED_PID_FILE is required}"
LAUNCHFILE="${G34_MATCHED_LAUNCHFILE:?G34_MATCHED_LAUNCHFILE is required}"
EPISODES="${G34_MATCHED_EPISODES:?G34_MATCHED_EPISODES is required}"
SEEDS_TEXT="${G34_MATCHED_SEEDS:?G34_MATCHED_SEEDS is required}"
ROS_PORT="${G34_MATCHED_ROS_PORT:?G34_MATCHED_ROS_PORT is required}"
GAZEBO_PORT="${G34_MATCHED_GAZEBO_PORT:?G34_MATCHED_GAZEBO_PORT is required}"
read -r -a SEEDS <<<"$SEEDS_TEXT"

FIVE_A="TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best"
EPOCH16="interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016"
DETECTOR="$BASE/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"
CHECKPOINT="$BASE/34_双头RewardAwareGate/local_data/training/seed20260912/best_runtime.pt"

stop_runtime() {
  local pgid children
  pgid="$(ps -o pgid= -p $$ | tr -d ' ')"
  children="$(ps -eo pid=,pgid= | awk -v p="$pgid" -v s="$$" '$2 == p && $1 != s {print $1}')"
  for child in $children; do kill -TERM "$child" 2>/dev/null || true; done
  children="$(ps -eo pid=,pgid= | awk -v p="$pgid" -v s="$$" '$2 == p && $1 != s {print $1}')"
  for child in $children; do kill -KILL "$child" 2>/dev/null || true; done
  fuser -k -KILL "${ROS_PORT}/tcp" "${GAZEBO_PORT}/tcp" >/dev/null 2>&1 || true
}
cleanup() { stop_runtime; unlink "$PID_FILE" 2>/dev/null || true; }
trap cleanup EXIT

exec 9>/tmp/local_critic_multi_robot_training.lock
echo "Waiting for single-Gazebo lock"
flock 9
echo "Acquired single-Gazebo lock"

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
export DRL_MULTI_TEST_TARGET_EPISODES="$EPISODES" DRL_MULTI_FIXED_PHYSICS_STEP_SIZE=0.001
export DRL_MULTI_REQUIRE_FIXED_STEP_SERVICE=1 DRL_MULTI_TEST_ACTOR_MODE=full
unset DRL_MULTI_CASE_ORACLE_MAP DRL_MULTI_RULE_ORACLE_MODE

verify_result() {
  python3 - "$1" "$MANIFEST" "$EPISODES" <<'PY'
import gzip,json,sys,numpy as np
rows=np.load(sys.argv[1],allow_pickle=True)
manifest=sys.argv[2]; episodes=int(sys.argv[3])
with gzip.open(manifest,'rt',encoding='utf-8') as handle:
    expected=[str(item['scenario_id']) for item in json.load(handle)['scenarios'][:episodes]]
if rows.shape != (episodes,17): raise SystemExit('wrong result shape')
if [str(item) for item in rows[:,12]] != expected: raise SystemExit('scenario order mismatch')
if sum(int(row[6])+int(row[7])+int(row[10]) for row in rows) != episodes*5:
    raise SystemExit('terminal accounting mismatch')
PY
}

verify_partial() {
  python3 - "$1" "$2" "$MANIFEST" "$EPISODES" <<'PY'
import gzip,json,sys,numpy as np,torch
rows=np.load(sys.argv[1],allow_pickle=True)
state=torch.load(sys.argv[2],map_location='cpu',weights_only=False)
target=int(sys.argv[4])
if rows.ndim != 2 or rows.shape[1] != 17 or not 0 < len(rows) < target:
    raise SystemExit('invalid partial result')
if len(set(rows[:,12].tolist())) != len(rows): raise SystemExit('duplicate scenarios')
with gzip.open(sys.argv[3],'rt',encoding='utf-8') as handle:
    expected=[str(item['scenario_id']) for item in json.load(handle)['scenarios'][:len(rows)]]
if [str(item) for item in rows[:,12]] != expected: raise SystemExit('partial order mismatch')
if int(state.get('episode_num',-1)) not in (len(rows),len(rows)+1):
    raise SystemExit('state/result mismatch')
print(len(rows))
PY
}

configure_g34() {
  export DRL_MULTI_STANDARD_ACTOR_FILE="$FIVE_A" DRL_MULTI_DENSE_ACTOR_FILE="$EPOCH16"
  export DRL_MULTI_DENSE_ACTOR_MODE=full DRL_MULTI_ACTOR_SELECTION_MODE=learned_gate
  export DRL_MULTI_GATE_DETECTOR_CHECKPOINT="$DETECTOR" DRL_MULTI_GATE_CHECKPOINT="$CHECKPOINT"
  export DRL_MULTI_GATE_SWITCH_ON_THRESHOLD=0.43 DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD=0.33
  export DRL_MULTI_GATE_MINIMUM_HOLD_STEPS=3 DRL_MULTI_GATE_EVALUATION_STRIDE=2
}

run_one() {
  local seed="$1" run_name="g34_g25slice_s${1}" result state log status progress
  result="$RESULT_DIR/${run_name}.npy"
  state="$STATE_DIR/${run_name}_state.pt"
  if [[ -f "$result" ]] && verify_result "$result" 2>/dev/null; then
    echo "Skipping completed $run_name"
    return 0
  fi
  export DRL_MULTI_SEED="$seed" DRL_MULTI_TEST_FILE_NAME="$run_name"
  export DRL_MULTI_TEST_STATS_PATH="$result" DRL_MULTI_TEST_STATE_PATH="$state"
  configure_g34
  for attempt in $(seq 1 10); do
    log="$LOG_DIR/${run_name}_attempt${attempt}.log"
    echo "Starting $run_name attempt $attempt"
    set +e
    (cd "$ROOT/TD3" && nice -n 10 python3 -u test_velodyne_td3_multi.py) >"$log" 2>&1
    status=$?
    set -e
    stop_runtime
    if [[ -f "$result" ]] && verify_result "$result" 2>/dev/null; then
      echo "Completed $run_name"
      return 0
    fi
    if [[ ! -f "$result" || ! -f "$state" ]]; then
      echo "$run_name failed before resumable state (exit=$status)" >&2
      return 1
    fi
    progress="$(verify_partial "$result" "$state")" || return 1
    echo "$run_name interrupted at ${progress}/${EPISODES}; exact resume"
  done
  echo "$run_name failed after 10 infrastructure attempts" >&2
  return 1
}

mkdir -p "$LOG_DIR" "$RESULT_DIR" "$STATE_DIR"
for seed in "${SEEDS[@]}"; do run_one "$seed"; done

python3 - "$MANIFEST" "$RESULT_DIR" "$RUN/local_data/matched/completion.json" "$EPISODES" "${SEEDS[*]}" <<'PY'
import hashlib,json,sys
from pathlib import Path
manifest,result_dir,output=Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3])
episodes=int(sys.argv[4]); seeds=[int(value) for value in sys.argv[5].split()]
hashes={}
for seed in seeds:
    path=result_dir/('g34_g25slice_s%d.npy' % seed)
    if not path.is_file(): raise SystemExit('missing result: %s' % path)
    hashes['g34_s%d' % seed]=hashlib.sha256(path.read_bytes()).hexdigest()
record={
    'format_version':1,
    'status':'complete',
    'experiment_id':'G34-G25-slice-matched-evaluation',
    'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
    'episodes_per_repeat':episodes,
    'seeds':seeds,
    'method':'g34',
    'total_episodes':episodes*len(seeds),
    'result_sha256':hashes,
    'actor_or_router_updated':False,
    'performance_early_stop':False,
}
output.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
PY
echo "G34 G25-slice matched evaluation complete"
