#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
SEALED="$BASE/25_最终消融与Sealed评测/local_data"
SOURCE_MANIFEST="$SEALED/sealed_manifest/dense_test_first256.json.gz"
SOURCE_RESULTS="$SEALED/sealed/results"
CAPTURE_ROOT="${TRAJECTORY_CAPTURE_ROOT:-$ROOT/paper/generated/captures/trajectory_capture}"
MANIFEST="$CAPTURE_ROOT/trajectory_subset_64.json.gz"
SELECTION="$CAPTURE_ROOT/trajectory_subset_selection.json"
RESULT_DIR="$CAPTURE_ROOT/results"
TRAJECTORY_DIR="$CAPTURE_ROOT/trajectories"
LOG_DIR="$ROOT/logs/active/trajectory-capture-overnight"
LAUNCHFILE="$LOG_DIR/runtime_trajectory_capture.launch"
PID_FILE="$ROOT/.trajectory_capture_queue.pid"
LOCK_FILE="/tmp/local_critic_multi_robot_training.lock"

EPISODES="${TRAJECTORY_CAPTURE_EPISODES:-64}"
SEED="${TRAJECTORY_CAPTURE_SEED:-20260910}"
METHODS_TEXT="${TRAJECTORY_CAPTURE_METHODS:-5a b2}"
read -r -a METHODS <<<"$METHODS_TEXT"
ROS_PORT="${TRAJECTORY_CAPTURE_ROS_PORT:-18623}"
GAZEBO_PORT="${TRAJECTORY_CAPTURE_GAZEBO_PORT:-18723}"
FIVE_A="TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best"
EPOCH16="interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016"
DETECTOR="$BASE/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"
B2="$BASE/11_可部署在线Gate研究/G11_B_student_rollout_v1/local_data/training/seed20260804/any/T1/best.pt"

mkdir -p "$CAPTURE_ROOT" "$RESULT_DIR" "$TRAJECTORY_DIR" "$LOG_DIR"

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

make_subset_manifest() {
  if [[ -f "$MANIFEST" && -f "$SELECTION" ]]; then
    echo "Using existing trajectory subset: $MANIFEST"
    return
  fi
  python - "$SOURCE_MANIFEST" "$SOURCE_RESULTS" "$MANIFEST" "$SELECTION" "$EPISODES" <<'PY'
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

source_manifest, source_results, output_manifest, output_selection, target = sys.argv[1:]
target = int(target)
with gzip.open(source_manifest, "rt", encoding="utf-8") as handle:
    payload = json.load(handle)
scenarios = payload["scenarios"]
ids = [str(item["scenario_id"]) for item in scenarios]
results = Path(source_results)
five_a = np.load(results / "g25_sealed_5a_s20260901.npy", allow_pickle=True)
b2 = np.load(results / "g25_sealed_b2_s20260901.npy", allow_pickle=True)
if five_a.shape != (256, 17) or b2.shape != (256, 17):
    raise SystemExit("sealed result shape is not (256, 17)")
if [str(x) for x in five_a[:, 12]] != ids or [str(x) for x in b2[:, 12]] != ids:
    raise SystemExit("sealed result order does not match manifest")

def metric(row, column):
    return float(row[column])

def density(index):
    metrics = scenarios[index].get("metrics", {})
    return (
        float(metrics.get("simultaneous_conflict_count", 0)),
        float(metrics.get("conflict_edge_count", 0)),
        float(metrics.get("interaction_density", 0)),
    )

five_full = five_a[:, 8].astype(int)
b2_full = b2[:, 8].astype(int)
groups = {
    "piroute_rescue": [i for i in range(256) if b2_full[i] == 1 and five_full[i] == 0],
    "both_success": [i for i in range(256) if b2_full[i] == 1 and five_full[i] == 1],
    "both_fail": [i for i in range(256) if b2_full[i] == 0 and five_full[i] == 0],
    "five_a_only_success": [i for i in range(256) if b2_full[i] == 0 and five_full[i] == 1],
}

# Keep equal-sized strata so the capture queue yields both explanatory successes
# and counterexamples rather than only the most favorable trajectories.
quotas = {
    "piroute_rescue": target // 4,
    "both_success": target // 4,
    "both_fail": target // 4,
    "five_a_only_success": target - 3 * (target // 4),
}
selected = []
selection_records = []
for group_name, quota in quotas.items():
    candidates = sorted(groups[group_name], key=density, reverse=True)
    chosen = candidates[:quota]
    if len(chosen) < quota:
        raise SystemExit("not enough scenes in group %s" % group_name)
    selected.extend(chosen)
    selection_records.append(
        {
            "group": group_name,
            "quota": quota,
            "scenario_ids": [ids[i] for i in chosen],
        }
    )

if len(selected) != target or len(set(selected)) != target:
    raise SystemExit("invalid trajectory subset selection")

subset = dict(payload)
subset["scenarios"] = [scenarios[i] for i in selected]
subset["split"] = "qualitative_trajectory_capture"
subset["trajectory_capture"] = {
    "source_manifest_sha256": hashlib.sha256(Path(source_manifest).read_bytes()).hexdigest(),
    "source_result_seed": 20260901,
    "selection_rule": "four outcome strata, highest conflict density within each stratum",
    "episode_count": target,
    "groups": selection_records,
}
Path(output_manifest).parent.mkdir(parents=True, exist_ok=True)
with gzip.open(output_manifest, "wt", encoding="utf-8") as handle:
    json.dump(subset, handle, ensure_ascii=False, indent=2)
Path(output_selection).write_text(
    json.dumps(
        {
            "source_manifest": source_manifest,
            "source_results": source_results,
            "selected_indices": selected,
            "selected_scenario_ids": [ids[i] for i in selected],
            "groups": selection_records,
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print("Wrote", output_manifest)
print("Selected", target, "scenes")
for record in selection_records:
    print(record["group"], len(record["scenario_ids"]))
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
  unlink "$PID_FILE" 2>/dev/null || true
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

verify_result() {
  python - "$1" "$MANIFEST" "$EPISODES" <<'PY'
import gzip
import json
import sys
from pathlib import Path
import numpy as np

result, manifest, target = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
rows = np.load(result, allow_pickle=True)
with gzip.open(manifest, "rt", encoding="utf-8") as handle:
    expected = [str(item["scenario_id"]) for item in json.load(handle)["scenarios"]]
observed = [str(item) for item in rows[:, 12]] if rows.ndim == 2 and rows.shape[1] > 12 else []
if rows.shape != (target, 17) or observed != expected:
    raise SystemExit("trajectory-capture result shape/order mismatch")
PY
}

configure_policy() {
  local method="$1"
  unset DRL_MULTI_DENSE_ACTOR_FILE DRL_MULTI_GATE_DETECTOR_CHECKPOINT
  unset DRL_MULTI_GATE_CHECKPOINT DRL_MULTI_GATE_SWITCH_ON_THRESHOLD
  unset DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD DRL_MULTI_GATE_MINIMUM_HOLD_STEPS
  unset DRL_MULTI_GATE_EVALUATION_STRIDE DRL_MULTI_ORACLE_INTERACTION_DISTANCE
  export DRL_MULTI_STANDARD_ACTOR_FILE="$FIVE_A"
  if [[ "$method" == "5a" ]]; then
    export DRL_MULTI_ACTOR_SELECTION_MODE=single
  elif [[ "$method" == "b2" ]]; then
    export DRL_MULTI_DENSE_ACTOR_FILE="$EPOCH16"
    export DRL_MULTI_ACTOR_SELECTION_MODE=learned_gate
    export DRL_MULTI_GATE_DETECTOR_CHECKPOINT="$DETECTOR"
    export DRL_MULTI_GATE_CHECKPOINT="$B2"
    export DRL_MULTI_GATE_SWITCH_ON_THRESHOLD=0.43
    export DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD=0.33
    export DRL_MULTI_GATE_MINIMUM_HOLD_STEPS=3
    export DRL_MULTI_GATE_EVALUATION_STRIDE=2
  else
    echo "unknown capture method: $method" >&2
    return 2
  fi
}

run_one() {
  local method="$1"
  local run_name="trajectory_capture_${method}_s${SEED}"
  local result="$RESULT_DIR/${run_name}.npy"
  local state="$RESULT_DIR/${run_name}_state.pt"
  local trajectory="$TRAJECTORY_DIR/${run_name}.jsonl"
  local log="$LOG_DIR/${run_name}.log"

  if [[ -f "$result" && -f "$trajectory" ]] && verify_result "$result" 2>/dev/null && [[ -s "$trajectory" ]]; then
    echo "Skipping completed $run_name"
    return 0
  fi

  rm -f "$result" "$state" "$trajectory"
  configure_policy "$method"
  export DRL_MULTI_SEED="$SEED"
  export DRL_MULTI_TEST_FILE_NAME="$run_name"
  export DRL_MULTI_TEST_STATE_PATH="$state"
  export DRL_MULTI_TEST_STATS_PATH="$result"
  export DRL_MULTI_TRAJECTORY_JSONL="$trajectory"
  export DRL_MULTI_TEST_TARGET_EPISODES="$EPISODES"
  export DRL_MULTI_SCENARIO=manifest
  export DRL_MULTI_MANIFEST_PATH="$MANIFEST"
  export DRL_MULTI_MANIFEST_SAMPLING=cycle
  export DRL_MULTI_NUM_AGENTS=5
  export DRL_MULTI_TEST_LAUNCHFILE="$LAUNCHFILE"
  export DRL_MULTI_TEST_ACTOR_MODE=full DRL_MULTI_DENSE_ACTOR_MODE=full
  export DRL_MULTI_FIXED_PHYSICS_STEP_SIZE=0.001
  export DRL_MULTI_REQUIRE_FIXED_STEP_SERVICE=1
  export DRL_MULTI_TRAJECTORY_INCLUDE_RAW_LIDAR="${TRAJECTORY_CAPTURE_INCLUDE_RAW_LIDAR:-0}"
  export DRL_MULTI_TRAJECTORY_INCLUDE_ROUTER_DIAGNOSTICS="${TRAJECTORY_CAPTURE_INCLUDE_ROUTER_DIAGNOSTICS:-0}"
  unset DRL_MULTI_CASE_ORACLE_MAP DRL_MULTI_RULE_ORACLE_MODE

  echo "Starting $run_name"
  wait_for_ports
  set +e
  (cd "$ROOT/TD3" && nice -n 10 python3 -u test_velodyne_td3_multi.py) >"$log" 2>&1
  local status=$?
  set -e
  stop_runtime_children
  wait_for_ports
  if ! verify_result "$result" 2>/dev/null || [[ ! -s "$trajectory" ]]; then
    echo "$run_name failed (exit=$status); inspect $log" >&2
    return 1
  fi
  echo "Completed $run_name"
}

make_subset_manifest
/usr/bin/python3 "$ROOT/scripts/generate_multi_robot_launch.py" --num-agents 5 --output "$LAUNCHFILE"

if [[ "${TRAJECTORY_CAPTURE_PREPARE_ONLY:-0}" == "1" ]]; then
  echo "Prepared trajectory capture manifest and launch file; queue not started"
  exit 0
fi

exec 9>"$LOCK_FILE"
echo "Waiting for the single-Gazebo lock"
flock 9
echo "Acquired the single-Gazebo lock"
echo "Trajectory capture queue: $EPISODES scenes x ${#METHODS[@]} frozen method(s)"
echo "Manifest: $MANIFEST"

for method in "${METHODS[@]}"; do
  run_one "$method"
done

python - "$CAPTURE_ROOT" "$MANIFEST" "$SEED" "$EPISODES" "$METHODS_TEXT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = Path(sys.argv[2])
seed = int(sys.argv[3])
episodes = int(sys.argv[4])
methods = sys.argv[5].split()
record = {
    "status": "complete",
    "purpose": "qualitative trajectory capture",
    "manifest": str(manifest),
    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "seed": seed,
    "methods": methods,
    "episodes_per_method": episodes,
    "sealed_statistics_modified": False,
}
(root / "trajectory_capture_completion.json").write_text(
    json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(root / "trajectory_capture_completion.json")
PY

echo "Trajectory capture queue complete"
