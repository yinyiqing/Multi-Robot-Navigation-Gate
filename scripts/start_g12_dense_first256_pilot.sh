#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
RUN_TAG="${G12_DENSE_RUN_TAG:-dense_first256_pilot}"
RUN_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/12_参数匹配单Actor容量对照/local_data/$RUN_TAG"
RESULTS_DIR="$RUN_DIR/results"
CHECKPOINT_DIR="$RUN_DIR/checkpoints"
LOG_DIR="$PROJECT_ROOT/logs/active/capacity-wide-g12-$RUN_TAG"
MANIFEST="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/dense/validation.json.gz"
LAUNCHFILE="multi_robot_scenario_fixed_v1_dense_validation_5d_5.launch"
BASE_MODEL="TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best"
INTERACTION_MODEL="interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016"
R2_MODEL="capacity_wide_r2_s4_broad_n5_seed20260816_epoch_001"
DETECTOR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"
B2_GATE="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究/G11_B_student_rollout_v1/local_data/training/seed20260804/any/T1/best.pt"
TARGET_EPISODES="${G12_DENSE_TARGET_EPISODES:-256}"
SEED="${G12_DENSE_SEED:-20260810}"

for required in \
  "$MANIFEST" \
  "$TD3_DIR/assets/$LAUNCHFILE" \
  "$TD3_DIR/pytorch_models/${BASE_MODEL}_actor.pth" \
  "$TD3_DIR/pytorch_models/${INTERACTION_MODEL}_actor.pth" \
  "$TD3_DIR/pytorch_models/${R2_MODEL}_actor.pth" \
  "$DETECTOR" \
  "$B2_GATE"; do
  [[ -f "$required" ]] || { echo "Required file is missing: $required" >&2; exit 1; }
done

if [[ "$(/usr/bin/python3 - "$MANIFEST" <<'PY'
import gzip
import json
import sys

with gzip.open(sys.argv[1], "rt", encoding="utf-8") as handle:
    payload = json.load(handle)
print(len(payload["scenarios"]))
PY
)" -lt "$TARGET_EPISODES" ]]; then
  echo "Dense manifest has fewer than $TARGET_EPISODES scenarios" >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR" "$CHECKPOINT_DIR" "$LOG_DIR"

labels=(5a epoch16 oracle b2 r2_10k)
ros_ports=(16001 16002 16003 16004 16005)
gazebo_ports=(16101 16102 16103 16104 16105)

for idx in "${!labels[@]}"; do
  label="${labels[$idx]}"
  pid_file="$PROJECT_ROOT/.g12_dense_${RUN_TAG}_${label}.pid"
  if [[ -f "$pid_file" ]]; then
    old_pid="$(tr -d '[:space:]' < "$pid_file")"
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "$label is already running with PID $old_pid" >&2
      exit 1
    fi
    unlink "$pid_file"
  fi
done

port_pattern=":(${ros_ports[*]// /|}|${gazebo_ports[*]// /|})$"
if ss -ltnH | awk '{print $4}' | rg -q "$port_pattern"; then
  echo "One or more dense pilot ROS/Gazebo ports are already in use" >&2
  exit 1
fi

start_one() {
  local label="$1"
  local ros_port="$2"
  local gazebo_port="$3"
  local run_name="g12_${RUN_TAG}_${label}_r1_s${SEED}"
  local pid_file="$PROJECT_ROOT/.g12_dense_${RUN_TAG}_${label}.pid"
  local state_path="$CHECKPOINT_DIR/${run_name}_state.pt"
  local stats_path="$RESULTS_DIR/${run_name}.npy"
  local log_file="$LOG_DIR/${run_name}.log"
  local runner_log="$LOG_DIR/${run_name}_runner.log"
  local policy_env

  case "$label" in
    5a)
      policy_env="
        export DRL_MULTI_STANDARD_ACTOR_FILE='$BASE_MODEL'
        export DRL_MULTI_ACTOR_SELECTION_MODE=single
        unset DRL_MULTI_DENSE_ACTOR_FILE DRL_MULTI_DENSE_ACTOR_MODE DRL_MULTI_GATE_CHECKPOINT DRL_MULTI_GATE_DETECTOR_CHECKPOINT
      "
      ;;
    epoch16)
      policy_env="
        export DRL_MULTI_STANDARD_ACTOR_FILE='$INTERACTION_MODEL'
        export DRL_MULTI_ACTOR_SELECTION_MODE=single
        unset DRL_MULTI_DENSE_ACTOR_FILE DRL_MULTI_DENSE_ACTOR_MODE DRL_MULTI_GATE_CHECKPOINT DRL_MULTI_GATE_DETECTOR_CHECKPOINT
      "
      ;;
    oracle)
      policy_env="
        export DRL_MULTI_STANDARD_ACTOR_FILE='$BASE_MODEL'
        export DRL_MULTI_DENSE_ACTOR_FILE='$INTERACTION_MODEL'
        export DRL_MULTI_DENSE_ACTOR_MODE=full
        export DRL_MULTI_ACTOR_SELECTION_MODE=interaction_oracle
        export DRL_MULTI_ORACLE_INTERACTION_DISTANCE=2.0
        unset DRL_MULTI_GATE_CHECKPOINT DRL_MULTI_GATE_DETECTOR_CHECKPOINT
      "
      ;;
    b2)
      policy_env="
        export DRL_MULTI_STANDARD_ACTOR_FILE='$BASE_MODEL'
        export DRL_MULTI_DENSE_ACTOR_FILE='$INTERACTION_MODEL'
        export DRL_MULTI_DENSE_ACTOR_MODE=full
        export DRL_MULTI_ACTOR_SELECTION_MODE=learned_gate
        export DRL_MULTI_GATE_DETECTOR_CHECKPOINT='$DETECTOR'
        export DRL_MULTI_GATE_CHECKPOINT='$B2_GATE'
        export DRL_MULTI_GATE_SWITCH_ON_THRESHOLD=0.43
        export DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD=0.33
        export DRL_MULTI_GATE_EVALUATION_STRIDE=2
        export DRL_MULTI_GATE_MINIMUM_HOLD_STEPS=3
      "
      ;;
    r2_10k)
      policy_env="
        export DRL_MULTI_STANDARD_ACTOR_FILE='$R2_MODEL'
        export DRL_MULTI_ACTOR_SELECTION_MODE=single
        unset DRL_MULTI_DENSE_ACTOR_FILE DRL_MULTI_DENSE_ACTOR_MODE DRL_MULTI_GATE_CHECKPOINT DRL_MULTI_GATE_DETECTOR_CHECKPOINT
      "
      ;;
    *)
      echo "Unknown pilot policy: $label" >&2
      return 2
      ;;
  esac

  setsid bash -lc "
    set -eo pipefail
    cleanup() {
      pgid=\"\$(ps -o pgid= -p \$\$ | tr -d ' ')\"
      ps -eo pid=,pgid= | awk -v pgid=\"\$pgid\" -v self=\"\$\$\" \\
        '\$2 == pgid && \$1 != self { print \$1 }' | xargs -r kill 2>/dev/null || true
      unlink '$pid_file' 2>/dev/null || true
    }
    trap cleanup EXIT
    source /opt/ros/noetic/setup.bash
    source '$PROJECT_ROOT/env.python.sh'
    source '$PROJECT_ROOT/catkin_ws/devel_isolated/setup.bash'
    export CUDA_VISIBLE_DEVICES=\"\"
    export ROS_HOSTNAME=localhost
    export ROS_MASTER_URI=http://localhost:$ros_port
    export ROS_PORT_SIM=$ros_port
    export GAZEBO_MASTER_URI=http://localhost:$gazebo_port
    export GAZEBO_IP=127.0.0.1
    export GAZEBO_RESOURCE_PATH='$PROJECT_ROOT/catkin_ws/src/multi_robot_scenario/launch'
    export DRL_MULTI_NUM_AGENTS=5
    export DRL_MULTI_SEED=$SEED
    export DRL_MULTI_TEST_FILE_NAME='$run_name'
    export DRL_MULTI_TEST_LAUNCHFILE='$LAUNCHFILE'
    export DRL_MULTI_TEST_ACTOR_MODE=full
    export DRL_MULTI_TEST_TARGET_EPISODES=$TARGET_EPISODES
    export DRL_MULTI_SCENARIO=manifest
    export DRL_MULTI_MANIFEST_PATH='$MANIFEST'
    export DRL_MULTI_MANIFEST_SAMPLING=cycle
    export DRL_MULTI_FIXED_PHYSICS_STEP_SIZE=0.001
    export DRL_MULTI_REQUIRE_FIXED_STEP_SERVICE=1
    export DRL_MULTI_TEST_STATE_PATH='$state_path'
    export DRL_MULTI_TEST_STATS_PATH='$stats_path'
    unset DRL_MULTI_CASE_ORACLE_MAP DRL_MULTI_RULE_ORACLE_MODE
    $policy_env
    cd '$TD3_DIR'
    python3 -u test_velodyne_td3_multi.py >'$log_file' 2>&1
  " >"$runner_log" 2>&1 < /dev/null &

  echo $! > "$pid_file"
  echo "$label PID: $(cat "$pid_file")"
  echo "$label log: $log_file"
}

for idx in "${!labels[@]}"; do
  start_one "${labels[$idx]}" "${ros_ports[$idx]}" "${gazebo_ports[$idx]}"
done

echo "Started dense run $RUN_TAG with seed $SEED."
echo "All policies use the first $TARGET_EPISODES scenarios from the frozen dense validation manifest."
