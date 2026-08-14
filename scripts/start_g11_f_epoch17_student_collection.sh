#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$ROOT/TD3"
GATE_ROOT="$ROOT/experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究"
RUN_DIR="$GATE_ROOT/G11_F_epoch17_gate_v1"
VIEW_DIR="$ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/g11_a1_gate_v1"
MANIFEST="$VIEW_DIR/train.json.gz"
BASE_MODEL="TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best"
INTERACTION_MODEL="avoidance_actor_from_5a_balanced_continue_e20_s20260813_best"
DETECTOR="$GATE_ROOT/../results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"
GATE="$RUN_DIR/local_data/a1_training/seed20260804/any/T1/best.pt"
RUN_METADATA="$RUN_DIR/student_run_metadata.json"
LOG_DIR="$ROOT/logs/active/g11_f_epoch17_gate"
PROFILE="${1:-}"

case "$PROFILE" in
  smoke)
    TARGET_EPISODES=1; ROS_PORT=16223; GAZEBO_PORT=16323
    OUTPUT_DIR="$RUN_DIR/local_data/smoke/student_shards"
    RUNTIME_DIR="$RUN_DIR/local_data/smoke/runtime"
    ;;
  train)
    TARGET_EPISODES=640; ROS_PORT=16423; GAZEBO_PORT=16523
    OUTPUT_DIR="$RUN_DIR/local_data/student_shards/train"
    RUNTIME_DIR="$RUN_DIR/local_data/runtime/train"
    ;;
  *) echo "Usage: $0 <smoke|train>" >&2; exit 2 ;;
esac

declare -A EXPECTED_SHA=(
  ["$MANIFEST"]="a97534ed22d1b4b12951cd42b80d515037774830f38cb432926c0e9f50379026"
  ["$TD3_DIR/pytorch_models/${BASE_MODEL}_actor.pth"]="fa28855049b67b3ee44c66d55d4f14441fc7c521e5429862c75b152f7d5cacc5"
  ["$TD3_DIR/pytorch_models/${INTERACTION_MODEL}_actor.pth"]="149c2e42848ecc9bc478cbed7fd89b9062936dbd5c669b55e6964441685155a5"
  ["$DETECTOR"]="0b914c0d090bbaba0a2be63c0d75d88580bf4d71778a7f1749c63989e3dbbd56"
  ["$GATE"]="b28e81d341c145d6fa8c881dd98c7ece5285231e7d080b3f71afcd2dfe3a0beb"
  ["$RUN_METADATA"]="1ef3cd4c0ef45f01255be62ed04ac284ac0713b8f1cb6a0f23da95499e06d313"
)
for path in "${!EXPECTED_SHA[@]}"; do
  [[ -f "$path" ]] || { echo "Missing frozen input: $path" >&2; exit 1; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "${EXPECTED_SHA[$path]}" ]] || {
    echo "Frozen input SHA-256 mismatch: $path" >&2; exit 1
  }
done

PID_FILE="$ROOT/.g11_f_epoch17_student_${PROFILE}.pid"
for candidate in "$ROOT/.g11_f_epoch17_student_smoke.pid" "$ROOT/.g11_f_epoch17_student_train.pid"; do
  [[ -f "$candidate" ]] || continue
  pid="$(tr -d '[:space:]' <"$candidate")"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    echo "G11-F student collection already runs as PID $pid" >&2; exit 1
  fi
  unlink "$candidate"
done
if pgrep -af '^python3(\.8)? -u (train|test)_velodyne_td3_multi.py($| )' >/dev/null; then
  echo "Another multi-robot training or evaluation process is running" >&2; exit 1
fi
for port in "$ROS_PORT" "$GAZEBO_PORT"; do
  ss -ltnH | awk '{print $4}' | grep -Eq ":${port}$" && {
    echo "Port $port is already in use" >&2; exit 1
  }
done

mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$RUNTIME_DIR/checkpoints" "$RUNTIME_DIR/results"
existing="$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.npz' | wc -l)"
(( existing < TARGET_EPISODES )) || {
  echo "G11-F $PROFILE already has $existing/$TARGET_EPISODES shards" >&2; exit 1
}
RUN_ID="g11_f_epoch17_student_${PROFILE}"
LOG_FILE="$LOG_DIR/${RUN_ID}_$(date +%Y%m%d_%H%M%S).log"
STATE_PATH="$RUNTIME_DIR/checkpoints/${RUN_ID}_state.pt"
STATS_PATH="$RUNTIME_DIR/results/${RUN_ID}.npy"

setsid bash -lc "
  set -eo pipefail
  cleanup() {
    pgid=\"\$(ps -o pgid= -p \$\$ | tr -d ' ')\"
    ps -eo pid=,pgid= | awk -v pgid=\"\$pgid\" -v self=\"\$\$\" \\
      '\$2 == pgid && \$1 != self { print \$1 }' | xargs -r kill 2>/dev/null || true
    unlink '$PID_FILE' 2>/dev/null || true
  }
  trap cleanup EXIT
  exec 9>/tmp/local_critic_multi_robot_training.lock
  flock -n 9 || { echo 'Multi-robot evaluation lock is busy' >&2; exit 1; }
  source /opt/ros/noetic/setup.bash
  source '$ROOT/env.python.sh'
  source '$ROOT/catkin_ws/devel_isolated/setup.bash'
  set -u
  export CUDA_VISIBLE_DEVICES=''
  export ROS_HOSTNAME=localhost ROS_MASTER_URI=http://localhost:$ROS_PORT ROS_PORT_SIM=$ROS_PORT
  export GAZEBO_MASTER_URI=http://localhost:$GAZEBO_PORT
  export GAZEBO_RESOURCE_PATH='$ROOT/catkin_ws/src/multi_robot_scenario/launch'
  export DRL_MULTI_NUM_AGENTS=5 DRL_MULTI_SEED=20260814
  export DRL_MULTI_TEST_LAUNCHFILE=multi_robot_scenario_strong_interaction_pilot_5.launch
  export DRL_MULTI_SCENARIO=manifest DRL_MULTI_MANIFEST_PATH='$MANIFEST' DRL_MULTI_MANIFEST_SAMPLING=cycle
  export DRL_MULTI_TEST_FILE_NAME='$RUN_ID' DRL_MULTI_STANDARD_ACTOR_FILE='$BASE_MODEL'
  export DRL_MULTI_DENSE_ACTOR_FILE='$INTERACTION_MODEL' DRL_MULTI_ACTOR_SELECTION_MODE=learned_gate
  export DRL_MULTI_TEST_ACTOR_MODE=full DRL_MULTI_DENSE_ACTOR_MODE=full
  export DRL_MULTI_GATE_DETECTOR_CHECKPOINT='$DETECTOR' DRL_MULTI_GATE_CHECKPOINT='$GATE'
  export DRL_MULTI_GATE_SWITCH_ON_THRESHOLD=0.29 DRL_MULTI_GATE_SWITCH_OFF_THRESHOLD=0.19
  export DRL_MULTI_GATE_MINIMUM_HOLD_STEPS=3 DRL_MULTI_GATE_EVALUATION_STRIDE=2
  export DRL_MULTI_TEST_TARGET_EPISODES=$TARGET_EPISODES
  export DRL_MULTI_TEST_STATE_PATH='$STATE_PATH' DRL_MULTI_TEST_STATS_PATH='$STATS_PATH'
  export DRL_MULTI_RAW_LIDAR_VOXEL_SIZE=0.01 DRL_MULTI_RAW_LIDAR_MAX_RANGE=6.0
  export DRL_MULTI_ROBOT_PERCEPTION_OUTPUT_DIR='$OUTPUT_DIR' DRL_MULTI_ROBOT_PERCEPTION_SPLIT=train
  export DRL_MULTI_ROBOT_PERCEPTION_FRAME_STRIDE=2 DRL_MULTI_ROBOT_PERCEPTION_MAX_BACKGROUND=12
  export DRL_MULTI_ROBOT_PERCEPTION_RUN_METADATA_PATH='$RUN_METADATA'
  cd '$TD3_DIR'
  nice -n 10 python3 -u test_velodyne_td3_multi.py
" >"$LOG_FILE" 2>&1 < /dev/null &

echo $! >"$PID_FILE"
echo "Started G11-F epoch-17 $PROFILE student collection"
echo "PID: $(cat "$PID_FILE")"
echo "Scenarios: $TARGET_EPISODES"
echo "Log: $LOG_FILE"
echo "Shards: $OUTPUT_DIR"
