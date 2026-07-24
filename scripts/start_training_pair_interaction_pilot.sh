#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
DATASET_DIR="$PROJECT_ROOT/experiments/04_保留专门化/05_论文主线/datasets/pair_interaction_curriculum_v1"
LOG_DIR="$PROJECT_ROOT/logs"
PID_FILE="$PROJECT_ROOT/.train_pair_interaction_pilot.pid"
BASE_MODEL="TD3_velodyne_multi_v4_curriculum_stage2_to_5d_geo_critic_from_5a_guarded_best"
MODEL_NAME="pair_interaction_pilot_s20260724"
LAUNCHFILE="multi_robot_scenario_multi_2.launch"
ROS_PORT="${DRL_MULTI_ROS_PORT:-12603}"
GAZEBO_PORT="${DRL_MULTI_GAZEBO_PORT:-12703}"

for path in "$DATASET_DIR/train.json.gz" "$DATASET_DIR/validation.json.gz"; do
  [[ -f "$path" ]] || { echo "Fixed pair-interaction split is missing: $path"; exit 1; }
done
for suffix in actor critic; do
  [[ -f "$TD3_DIR/pytorch_models/${BASE_MODEL}_${suffix}.pth" ]] || {
    echo "5D ${suffix} is missing: $TD3_DIR/pytorch_models/${BASE_MODEL}_${suffix}.pth"
    exit 1
  }
done
[[ -f "$TD3_DIR/assets/$LAUNCHFILE" ]] || {
  echo "Two-agent launch file is missing: $TD3_DIR/assets/$LAUNCHFILE"
  exit 1
}

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Pair-interaction pilot is already running with PID $old_pid"
    exit 1
  fi
  unlink "$PID_FILE"
fi
if [[ -e "$TD3_DIR/checkpoints/${MODEL_NAME}_latest.pt" ]]; then
  echo "Pilot checkpoint already exists; archive it before starting a fresh run."
  exit 1
fi
if ss -ltn | awk '{print $4}' | grep -Eq ":${ROS_PORT}$"; then
  echo "ROS port $ROS_PORT is already in use."
  exit 1
fi
if ss -ltn | awk '{print $4}' | grep -Eq ":${GAZEBO_PORT}$"; then
  echo "Gazebo port $GAZEBO_PORT is already in use."
  exit 1
fi

mkdir -p "$LOG_DIR"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$LOG_DIR/train_pair_interaction_pilot_${timestamp}.log"

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
  export ROS_HOSTNAME=localhost
  export ROS_MASTER_URI=http://localhost:$ROS_PORT
  export ROS_PORT_SIM=$ROS_PORT
  export GAZEBO_MASTER_URI=http://localhost:$GAZEBO_PORT
  export GAZEBO_RESOURCE_PATH='$PROJECT_ROOT/catkin_ws/src/multi_robot_scenario/launch'
  export DRL_MULTI_NUM_AGENTS=2
  export DRL_MULTI_SEED=20260724
  export DRL_MULTI_TRAIN_LAUNCHFILE='$LAUNCHFILE'
  export DRL_MULTI_SCENARIO=manifest
  export DRL_MULTI_MANIFEST_PATH='$DATASET_DIR/train.json.gz'
  export DRL_MULTI_EVAL_MANIFEST_PATH='$DATASET_DIR/validation.json.gz'
  export DRL_MULTI_MANIFEST_SAMPLING=random
  export DRL_MULTI_TRAIN_FILE_NAME='$MODEL_NAME'
  export DRL_MULTI_LOAD_MODEL=1
  export DRL_MULTI_LOAD_ACTOR_ONLY=0
  export DRL_MULTI_LOAD_MODEL_NAME='$BASE_MODEL'
  export DRL_MULTI_RESUME_TRAINING=0
  export DRL_MULTI_MAX_EPOCHS=2
  export DRL_MULTI_EVAL_EPISODES=30
  export DRL_MULTI_EVAL_FREQ_AGENT_SAMPLES=12000
  export DRL_MULTI_BEST_METRIC=full_success
  export DRL_MULTI_TRAINING_VERSION='pair-interaction-pilot-v1'
  export DRL_MULTI_ACTOR_TRAIN_MODE=full
  export DRL_MULTI_USE_DYNAMIC_REWARD=1
  export DRL_MULTI_REWARD_MODE=average
  export DRL_MULTI_REWARD_SELF_WEIGHT=0.8
  export DRL_MULTI_USE_DISTANCE_WEIGHTED_REWARD=1
  export DRL_MULTI_REWARD_SIGMA=2.0
  export DRL_MULTI_USE_LOCAL_CRITIC=1
  export DRL_MULTI_LOCAL_CRITIC_GEOMETRY_ONLY=1
  export DRL_MULTI_LOCAL_CRITIC_MAX_AGENTS=10
  export DRL_MULTI_ACTIVE_NEIGHBORS_ONLY=1
  export DRL_MULTI_FORWARD_REWARD_WEIGHT=0.0
  export DRL_MULTI_STAGNATION_PENALTY_WEIGHT=0.0
  export DRL_MULTI_USE_ANTI_STAGNATION_REWARD=0
  export DRL_MULTI_USE_LOCAL_NAVIGATION_REWARD=0
  export DRL_MULTI_USE_WALL_CLEARANCE_REWARD=0
  export DRL_MULTI_EXPL_NOISE=0.05
  export DRL_MULTI_EXPL_MIN=0.02
  export DRL_MULTI_EXPL_DECAY_STEPS=50000
  export DRL_MULTI_ACTOR_LR=0.000001
  export DRL_MULTI_CRITIC_LR=0.00008
  export DRL_MULTI_ACTOR_ANCHOR_WEIGHT=0.0
  export DRL_MULTI_ACTOR_Q_NORMALIZATION_ALPHA=0.0
  export DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS=13000
  export DRL_MULTI_POLICY_FREQ=2
  cd '$TD3_DIR'
  python3 -u train_velodyne_td3_multi.py
" >"$log_file" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "Started pair-interaction pilot."
echo "PID: $(cat "$PID_FILE")"
echo "Model: $MODEL_NAME"
echo "Warm start: full 5D Actor/Critic"
echo "Architecture: original TD3 Actor (24 -> 800 -> 600 -> 2)"
echo "Training: 90 fixed random pair scenarios"
echo "Validation: 30 fixed held-out pair scenarios"
echo "Epoch 1: frozen Actor baseline; Epoch 2: full Actor training"
echo "Log: $log_file"
echo "Expected runtime: roughly 2-4 hours."
