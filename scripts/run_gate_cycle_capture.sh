#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
G25="$BASE/25_最终消融与Sealed评测/local_data"
SOURCE_MANIFEST="$G25/sealed_manifest/dense_test_first256.json.gz"
SOURCE_RESULTS="$G25/sealed/results"
CAPTURE_ROOT="${GATE_CYCLE_CAPTURE_ROOT:-$ROOT/paper/generated/captures/gate_cycle_capture_corrected}"
MANIFEST_DIR="$CAPTURE_ROOT/manifests"
RESULT_DIR="$CAPTURE_ROOT/results"
TRAJECTORY_DIR="$CAPTURE_ROOT/trajectories"
LOG_DIR="$ROOT/logs/active/gate-cycle-capture"
LAUNCHFILE="$LOG_DIR/runtime_gate_cycle_capture.launch"
LOCK_FILE="/tmp/local_critic_multi_robot_training.lock"

ROS_PORT="${GATE_CYCLE_CAPTURE_ROS_PORT:-18823}"
GAZEBO_PORT="${GATE_CYCLE_CAPTURE_GAZEBO_PORT:-18923}"
FIVE_A="TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best"
EPOCH16="interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016"
DETECTOR="$BASE/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"
B2="$BASE/11_可部署在线Gate研究/G11_B_student_rollout_v1/local_data/training/seed20260804/any/T1/best.pt"
SEEDS=(20260901 20260902 20260903)

mkdir -p "$CAPTURE_ROOT" "$MANIFEST_DIR" "$RESULT_DIR" "$TRAJECTORY_DIR" "$LOG_DIR"

set +u
source /opt/ros/noetic/setup.bash
source "$ROOT/env.python.sh"
source "$ROOT/catkin_ws/devel_isolated/setup.bash"
set -u

export CUDA_VISIBLE_DEVICES=""
export ROS_HOSTNAME=localhost
export ROS_MASTER_URI="http://localhost:$ROS_PORT"
export ROS_PORT_SIM="$ROS_PORT"
export GAZEBO_MASTER_URI="http://localhost:$GAZEBO_PORT"
export GAZEBO_IP=127.0.0.1
export GAZEBO_RESOURCE_PATH="$ROOT/catkin_ws/src/multi_robot_scenario/launch"

make_manifests() {
  python - "$SOURCE_MANIFEST" "$SOURCE_RESULTS" "$MANIFEST_DIR" "$CAPTURE_ROOT/selection.json" <<'PY'
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

source_manifest, source_results, manifest_dir, selection_path = sys.argv[1:]
source_results = Path(source_results)
manifest_dir = Path(manifest_dir)
with gzip.open(source_manifest, "rt", encoding="utf-8") as handle:
    payload = json.load(handle)
scenarios = payload["scenarios"]
scenario_by_id = {str(item["scenario_id"]): item for item in scenarios}
expected_ids = [str(item["scenario_id"]) for item in scenarios]

selection = {
    "purpose": "post-sealed qualitative diagnosis of high-switch B2 episodes",
    "selection_rule": "all G25 B2 sealed scene/repeat rows with episode gate switches > 10",
    "source_manifest": source_manifest,
    "source_manifest_sha256": hashlib.sha256(Path(source_manifest).read_bytes()).hexdigest(),
    "sealed_statistics_modified": False,
    "groups": [],
}
for seed in (20260901, 20260902, 20260903):
    result_path = source_results / f"g25_sealed_b2_s{seed}.npy"
    rows = np.load(result_path, allow_pickle=True)
    if rows.shape != (256, 17):
        raise SystemExit(f"unexpected sealed result shape: {result_path}: {rows.shape}")
    observed_ids = [str(value) for value in rows[:, 12]]
    if observed_ids != expected_ids:
        raise SystemExit(f"sealed result order does not match manifest: {result_path}")
    selected_rows = [row for row in rows if int(row[15]) > 10]
    selected_ids = [str(row[12]) for row in selected_rows]
    subset = dict(payload)
    subset["split"] = "qualitative_gate_cycle_capture"
    subset["scenarios"] = [scenario_by_id[scenario_id] for scenario_id in selected_ids]
    subset["gate_cycle_capture"] = {
        "source_result": str(result_path),
        "source_result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "repeat_seed": seed,
        "selection_rule": selection["selection_rule"],
    }
    output = manifest_dir / f"high_switch_s{seed}.json.gz"
    with gzip.open(output, "wt", encoding="utf-8") as handle:
        json.dump(subset, handle, ensure_ascii=False, indent=2)
    group = {
        "seed": seed,
        "manifest": str(output),
        "episodes": len(selected_rows),
        "rows": [
            {
                "scenario_id": str(row[12]),
                "sealed_episode_steps": int(row[3]),
                "sealed_full_success": int(row[8]),
                "sealed_interaction_share": float(row[14]),
                "sealed_gate_switches": int(row[15]),
            }
            for row in selected_rows
        ],
    }
    selection["groups"].append(group)

if sum(group["episodes"] for group in selection["groups"]) != 9:
    raise SystemExit("expected exactly nine high-switch scene/repeat pairs")
Path(selection_path).write_text(
    json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(selection_path)
for group in selection["groups"]:
    print(group["seed"], group["episodes"], group["manifest"])
PY
}

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
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_ports() {
  for _ in $(seq 1 60); do
    if ! ss -ltnH | awk '{print $4}' | rg -q ":${ROS_PORT}$|:${GAZEBO_PORT}$"; then
      return 0
    fi
    sleep 1
  done
  echo "ROS/Gazebo ports did not become free" >&2
  return 1
}

episode_count() {
  python - "$1" <<'PY'
import gzip
import json
import sys
with gzip.open(sys.argv[1], "rt", encoding="utf-8") as handle:
    print(len(json.load(handle)["scenarios"]))
PY
}

verify_result() {
  python - "$1" "$2" <<'PY'
import gzip
import json
import sys
import numpy as np

result, manifest = sys.argv[1:]
rows = np.load(result, allow_pickle=True)
with gzip.open(manifest, "rt", encoding="utf-8") as handle:
    expected = [str(item["scenario_id"]) for item in json.load(handle)["scenarios"]]
observed = [str(item) for item in rows[:, 12]] if rows.ndim == 2 and rows.shape[1] > 12 else []
if rows.shape != (len(expected), 17) or observed != expected:
    raise SystemExit("gate-cycle capture result shape/order mismatch")
PY
}

configure_b2() {
  export DRL_MULTI_STANDARD_ACTOR_FILE="$FIVE_A"
  export DRL_MULTI_DENSE_ACTOR_FILE="$EPOCH16"
  export DRL_MULTI_ACTOR_SELECTION_MODE=learned_gate
  export DRL_MULTI_GATE_DETECTOR_CHECKPOINT="$DETECTOR"
  export DRL_MULTI_GATE_CHECKPOINT="$B2"
  export DRL_MULTI_GATE_SWITCH_ON_THRESHOLD=0.43
  export DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD=0.33
  export DRL_MULTI_GATE_MINIMUM_HOLD_STEPS=3
  export DRL_MULTI_GATE_EVALUATION_STRIDE=2
}

run_seed() {
  local seed="$1"
  local manifest="$MANIFEST_DIR/high_switch_s${seed}.json.gz"
  local episodes
  episodes="$(episode_count "$manifest")"
  local run_name="gate_cycle_capture_b2_s${seed}"
  local result="$RESULT_DIR/${run_name}.npy"
  local state="$RESULT_DIR/${run_name}_state.pt"
  local trajectory="$TRAJECTORY_DIR/${run_name}.jsonl"
  local log="$LOG_DIR/${run_name}.log"

  if [[ -f "$result" && -s "$trajectory" ]] && verify_result "$result" "$manifest" 2>/dev/null; then
    echo "Skipping completed $run_name"
    return 0
  fi

  rm -f "$result" "$state" "$trajectory"
  configure_b2
  export DRL_MULTI_SEED="$seed"
  export DRL_MULTI_TEST_FILE_NAME="$run_name"
  export DRL_MULTI_TEST_STATE_PATH="$state"
  export DRL_MULTI_TEST_STATS_PATH="$result"
  export DRL_MULTI_TRAJECTORY_JSONL="$trajectory"
  export DRL_MULTI_TEST_TARGET_EPISODES="$episodes"
  export DRL_MULTI_SCENARIO=manifest
  export DRL_MULTI_MANIFEST_PATH="$manifest"
  export DRL_MULTI_MANIFEST_SAMPLING=cycle
  export DRL_MULTI_NUM_AGENTS=5
  export DRL_MULTI_TEST_LAUNCHFILE="$LAUNCHFILE"
  export DRL_MULTI_TEST_ACTOR_MODE=full
  export DRL_MULTI_DENSE_ACTOR_MODE=full
  export DRL_MULTI_FIXED_PHYSICS_STEP_SIZE=0.001
  export DRL_MULTI_REQUIRE_FIXED_STEP_SERVICE=1
  export DRL_MULTI_TRAJECTORY_INCLUDE_RAW_LIDAR=0
  export DRL_MULTI_TRAJECTORY_INCLUDE_ROUTER_DIAGNOSTICS=1
  unset DRL_MULTI_CASE_ORACLE_MAP DRL_MULTI_RULE_ORACLE_MODE

  echo "Starting $run_name ($episodes episodes)"
  wait_for_ports
  set +e
  (cd "$ROOT/TD3" && nice -n 10 python3 -u test_velodyne_td3_multi.py) >"$log" 2>&1
  local status=$?
  set -e
  stop_runtime_children
  wait_for_ports
  if [[ "$status" -ne 0 ]] || ! verify_result "$result" "$manifest" 2>/dev/null || [[ ! -s "$trajectory" ]]; then
    echo "$run_name failed (exit=$status); inspect $log" >&2
    return 1
  fi
  echo "Completed $run_name"
}

make_manifests
/usr/bin/python3 "$ROOT/scripts/generate_multi_robot_launch.py" --num-agents 5 --output "$LAUNCHFILE"

if [[ "${GATE_CYCLE_CAPTURE_PREPARE_ONLY:-0}" == "1" ]]; then
  echo "Prepared high-switch manifests and launch file; queue not started"
  exit 0
fi

exec 9>"$LOCK_FILE"
echo "Waiting for the single-Gazebo lock"
flock 9
echo "Acquired the single-Gazebo lock"
for seed in "${SEEDS[@]}"; do
  run_seed "$seed"
done

python "$ROOT/scripts/analyze_gate_cycle_capture.py" --root "$CAPTURE_ROOT"
echo "Gate-cycle capture queue complete"
