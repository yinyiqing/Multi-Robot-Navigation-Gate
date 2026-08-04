#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
VIEW_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/g11_a1_gate_v1"
RUN_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究/G11_B_student_rollout_v1"
LOG_DIR="$PROJECT_ROOT/logs/active/g11_b"
BASE_MODEL="TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best"
INTERACTION_MODEL="interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016"
DETECTOR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"
GATE="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究/G11_A1_当前协议时序pilot/local_data/training/seed20260804/any/T1/best.pt"
MANIFEST="$VIEW_DIR/train.json.gz"
RUN_METADATA="$RUN_DIR/student_run_metadata.json"
PROFILE="${1:-}"

case "$PROFILE" in
  smoke)
    TARGET_EPISODES=1
    ROS_PORT=14223
    GAZEBO_PORT=14323
    OUTPUT_DIR="$RUN_DIR/local_data/smoke/student_shards"
    RUNTIME_DIR="$RUN_DIR/local_data/smoke/runtime"
    ;;
  train)
    TARGET_EPISODES=640
    ROS_PORT=14423
    GAZEBO_PORT=14523
    OUTPUT_DIR="$RUN_DIR/local_data/student_shards/train"
    RUNTIME_DIR="$RUN_DIR/local_data/runtime/train"
    ;;
  *)
    echo "Usage: $0 <smoke|train>" >&2
    exit 2
    ;;
esac

declare -A EXPECTED_SHA=(
  ["$MANIFEST"]="a97534ed22d1b4b12951cd42b80d515037774830f38cb432926c0e9f50379026"
  ["$TD3_DIR/pytorch_models/${BASE_MODEL}_actor.pth"]="fa28855049b67b3ee44c66d55d4f14441fc7c521e5429862c75b152f7d5cacc5"
  ["$TD3_DIR/pytorch_models/${INTERACTION_MODEL}_actor.pth"]="6ec1942fcd497ab1cc2a85a5aaec8f524395dc21ff21a442dca243a52e917c0b"
  ["$DETECTOR"]="0b914c0d090bbaba0a2be63c0d75d88580bf4d71778a7f1749c63989e3dbbd56"
  ["$GATE"]="d9b05d9f86e5bad4d2071c041187b618ebca6f1a3cc1f9c46e8b14b1a451537a"
  ["$RUN_METADATA"]="a9a3d1c38c26674dd44a2013c5e2172ce34d7623393990d3115b9d2228005a3c"
)
for path in "${!EXPECTED_SHA[@]}"; do
  [[ -f "$path" ]] || { echo "Required input is missing: $path" >&2; exit 1; }
  actual_sha="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual_sha" == "${EXPECTED_SHA[$path]}" ]] || {
    echo "Frozen input hash mismatch: $path" >&2
    exit 1
  }
done

PID_FILE="$PROJECT_ROOT/.g11_b_student_${PROFILE}.pid"
for candidate_pid_file in \
  "$PROJECT_ROOT/.g11_b_student_smoke.pid" \
  "$PROJECT_ROOT/.g11_b_student_train.pid"; do
  [[ -f "$candidate_pid_file" ]] || continue
  old_pid="$(tr -d '[:space:]' < "$candidate_pid_file")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "A G11-B collection is already running with PID $old_pid" >&2
    exit 1
  fi
  unlink "$candidate_pid_file"
done
for port in "$ROS_PORT" "$GAZEBO_PORT"; do
  if ss -ltnH | awk '{print $4}' | rg -q ":${port}$"; then
    echo "Port $port is already in use" >&2
    exit 1
  fi
done

mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$RUNTIME_DIR/checkpoints" "$RUNTIME_DIR/results"
existing_shards="$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.npz' | wc -l)"
if (( existing_shards >= TARGET_EPISODES )); then
  echo "G11-B $PROFILE already has $existing_shards/$TARGET_EPISODES shards" >&2
  exit 1
fi
timestamp="$(date +%Y%m%d_%H%M%S)"
run_id="g11_b_student_${PROFILE}"
log_file="$LOG_DIR/collect_${run_id}_${timestamp}.log"
state_path="$RUNTIME_DIR/checkpoints/${run_id}_state.pt"
stats_path="$RUNTIME_DIR/results/${run_id}.npy"

setsid bash -lc "
  set -eo pipefail
  cleanup() {
    pgid=\"\$(ps -o pgid= -p \$\$ | tr -d ' ')\"
    ps -eo pid=,pgid= | awk -v pgid=\"\$pgid\" -v self=\"\$\$\" \\
      '\$2 == pgid && \$1 != self { print \$1 }' | xargs -r kill 2>/dev/null || true
    unlink '$PID_FILE' 2>/dev/null || true
  }
  trap cleanup EXIT
  source /opt/ros/noetic/setup.bash
  source '$PROJECT_ROOT/env.python.sh'
  source '$PROJECT_ROOT/catkin_ws/devel_isolated/setup.bash'
  export CUDA_VISIBLE_DEVICES=''
  export ROS_HOSTNAME=localhost
  export ROS_MASTER_URI=http://localhost:$ROS_PORT
  export ROS_PORT_SIM=$ROS_PORT
  export GAZEBO_MASTER_URI=http://localhost:$GAZEBO_PORT
  export GAZEBO_RESOURCE_PATH='$PROJECT_ROOT/catkin_ws/src/multi_robot_scenario/launch'
  export DRL_MULTI_NUM_AGENTS=5
  export DRL_MULTI_SEED=20260804
  export DRL_MULTI_TEST_LAUNCHFILE='multi_robot_scenario_strong_interaction_pilot_5.launch'
  export DRL_MULTI_SCENARIO=manifest
  export DRL_MULTI_MANIFEST_PATH='$MANIFEST'
  export DRL_MULTI_MANIFEST_SAMPLING=cycle
  export DRL_MULTI_TEST_FILE_NAME='$run_id'
  export DRL_MULTI_STANDARD_ACTOR_FILE='$BASE_MODEL'
  export DRL_MULTI_DENSE_ACTOR_FILE='$INTERACTION_MODEL'
  export DRL_MULTI_ACTOR_SELECTION_MODE=learned_gate
  export DRL_MULTI_TEST_ACTOR_MODE=full
  export DRL_MULTI_DENSE_ACTOR_MODE=full
  export DRL_MULTI_GATE_DETECTOR_CHECKPOINT='$DETECTOR'
  export DRL_MULTI_GATE_CHECKPOINT='$GATE'
  export DRL_MULTI_GATE_SWITCH_ON_THRESHOLD=0.28
  export DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD=0.18
  export DRL_MULTI_GATE_MINIMUM_HOLD_STEPS=3
  export DRL_MULTI_GATE_EVALUATION_STRIDE=2
  export DRL_MULTI_TEST_TARGET_EPISODES=$TARGET_EPISODES
  export DRL_MULTI_TEST_STATE_PATH='$state_path'
  export DRL_MULTI_TEST_STATS_PATH='$stats_path'
  export DRL_MULTI_RAW_LIDAR_VOXEL_SIZE=0.01
  export DRL_MULTI_RAW_LIDAR_MAX_RANGE=6.0
  export DRL_MULTI_ROBOT_PERCEPTION_OUTPUT_DIR='$OUTPUT_DIR'
  export DRL_MULTI_ROBOT_PERCEPTION_SPLIT=train
  export DRL_MULTI_ROBOT_PERCEPTION_FRAME_STRIDE=2
  export DRL_MULTI_ROBOT_PERCEPTION_MAX_BACKGROUND=12
  export DRL_MULTI_ROBOT_PERCEPTION_RUN_METADATA_PATH='$RUN_METADATA'
  cd '$TD3_DIR'
  nice -n 10 python3 -u test_velodyne_td3_multi.py
" >"$log_file" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "Started G11-B $PROFILE student collection."
echo "PID: $(cat "$PID_FILE")"
echo "Scenarios: $TARGET_EPISODES"
echo "Log: $log_file"
echo "Shards: $OUTPUT_DIR"
