#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
DATASET_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1"
TRAIN_MANIFEST="${DRL_MULTI_TRAIN_MANIFEST:-$DATASET_DIR/views/g12_r3_mixed_v1/train.json.gz}"
EVAL_MANIFEST="${DRL_MULTI_EVAL_MANIFEST:-$DATASET_DIR/views/g12_full_scene_selection_v1/validation.json.gz}"
BASE_MODEL="${DRL_MULTI_LOAD_MODEL_NAME:-TD3_velodyne_multi_v4_curriculum_stage2_to_3d2_geo_critic_from_3a_guarded_best}"
MODEL_NAME="${DRL_MULTI_TRAIN_FILE_NAME:-current_generalist_fullscene_local_critic_guarded_s20260809}"
TRAINING_VERSION="${DRL_MULTI_TRAINING_VERSION:-current-generalist-fullscene-local-critic-v1}"
EXPERIMENT_LABEL="${DRL_MULTI_EXPERIMENT_LABEL:-current-generalist-fullscene-local-critic}"
SAFE_MODEL="${MODEL_NAME//[^A-Za-z0-9_]/_}"
PID_FILE="${DRL_MULTI_PID_FILE:-$PROJECT_ROOT/.train_${SAFE_MODEL}.pid}"
LOG_DIR="${DRL_MULTI_LOG_DIR:-$PROJECT_ROOT/logs/active/$EXPERIMENT_LABEL}"
LAUNCHFILE="multi_robot_scenario_stage2_to_5a_shared_guarded_5.launch"
ROS_PORT="${DRL_MULTI_ROS_PORT:-14261}"
GAZEBO_PORT="${DRL_MULTI_GAZEBO_PORT:-14361}"
SEED="${DRL_MULTI_SEED:-20260808}"
MAX_EPOCHS="${DRL_MULTI_MAX_EPOCHS:-4}"
EVAL_EPISODES="${DRL_MULTI_EVAL_EPISODES:-120}"
EVAL_FREQ="${DRL_MULTI_EVAL_FREQ_AGENT_SAMPLES:-20000}"
ACTOR_UPDATE_DELAY="${DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS:-40000}"
ACTOR_ANCHOR_WEIGHT="${DRL_MULTI_ACTOR_ANCHOR_WEIGHT:-0.05}"
ACTOR_Q_NORM_ALPHA="${DRL_MULTI_ACTOR_Q_NORMALIZATION_ALPHA:-1.0}"
ACTOR_GRAD_CLIP="${DRL_MULTI_ACTOR_GRAD_NORM_CLIP:-1.0}"
RESUME_TRAINING="${DRL_MULTI_RESUME_TRAINING:-0}"

for path in "$TRAIN_MANIFEST" "$EVAL_MANIFEST"; do
  [[ -f "$path" ]] || { echo "Required manifest is missing: $path" >&2; exit 1; }
done
[[ -f "$TD3_DIR/pytorch_models/${BASE_MODEL}_actor.pth" ]] || {
  echo "Warm-start Actor is missing: $TD3_DIR/pytorch_models/${BASE_MODEL}_actor.pth" >&2
  exit 1
}
[[ -f "$TD3_DIR/assets/$LAUNCHFILE" ]] || {
  echo "Launch file is missing: $TD3_DIR/assets/$LAUNCHFILE" >&2
  exit 1
}
[[ "$RESUME_TRAINING" == 0 || "$RESUME_TRAINING" == 1 ]] || {
  echo "DRL_MULTI_RESUME_TRAINING must be 0 or 1" >&2
  exit 2
}
[[ "$ACTOR_UPDATE_DELAY" =~ ^[0-9]+$ ]] || {
  echo "DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS must be a non-negative integer" >&2
  exit 2
}
if [[ "$RESUME_TRAINING" == 0 && -e "$TD3_DIR/checkpoints/${MODEL_NAME}_latest.pt" ]]; then
  echo "Fresh-run checkpoint already exists: $TD3_DIR/checkpoints/${MODEL_NAME}_latest.pt" >&2
  echo "Use a new DRL_MULTI_TRAIN_FILE_NAME or set DRL_MULTI_RESUME_TRAINING=1 explicitly." >&2
  exit 1
fi
if [[ "$RESUME_TRAINING" == 1 && ! -e "$TD3_DIR/checkpoints/${MODEL_NAME}_latest.pt" ]]; then
  echo "Resume checkpoint is missing: $TD3_DIR/checkpoints/${MODEL_NAME}_latest.pt" >&2
  exit 1
fi

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Current generalist retrain is already running with PID $old_pid" >&2
    exit 1
  fi
  unlink "$PID_FILE"
fi

if pgrep -af '^python3(\.8)? -u train_velodyne_td3_multi\.py($| )' >/dev/null; then
  echo "Another multi-robot training process is already running" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$LOG_DIR/train_${SAFE_MODEL}_${timestamp}.log"
runner_log="$LOG_DIR/train_${SAFE_MODEL}_${timestamp}_runner.log"

setsid bash -lc "
  set -eo pipefail
  cleanup() {
    pgid=\"\$(ps -o pgid= -p \$\$ | tr -d ' ')\"
    ps -eo pid=,pgid= | awk -v pgid=\"\$pgid\" -v self=\"\$\$\" '\$2 == pgid && \$1 != self { print \$1 }' | xargs -r kill 2>/dev/null || true
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
  export GAZEBO_IP=127.0.0.1
  export GAZEBO_RESOURCE_PATH='$PROJECT_ROOT/catkin_ws/src/multi_robot_scenario/launch'

  export DRL_MULTI_NUM_AGENTS=5
  export DRL_MULTI_SEED='$SEED'
  export DRL_MULTI_TRAIN_LAUNCHFILE='$LAUNCHFILE'
  export DRL_MULTI_SCENARIO=manifest
  export DRL_MULTI_MANIFEST_PATH='$TRAIN_MANIFEST'
  export DRL_MULTI_EVAL_MANIFEST_PATH='$EVAL_MANIFEST'
  export DRL_MULTI_MANIFEST_SAMPLING=cycle

  export DRL_MULTI_TRAIN_FILE_NAME='$MODEL_NAME'
  export DRL_MULTI_TRAINING_VERSION='$TRAINING_VERSION'
  export DRL_MULTI_LOAD_MODEL=1
  export DRL_MULTI_LOAD_ACTOR_ONLY=1
  export DRL_MULTI_REQUIRE_MODEL_LOAD=1
  export DRL_MULTI_LOAD_MODEL_NAME='$BASE_MODEL'
  export DRL_MULTI_RESUME_TRAINING='$RESUME_TRAINING'
  export DRL_MULTI_MAX_EPOCHS='$MAX_EPOCHS'
  export DRL_MULTI_EVAL_EPISODES='$EVAL_EPISODES'
  export DRL_MULTI_EVAL_FREQ_AGENT_SAMPLES='$EVAL_FREQ'
  export DRL_MULTI_BEST_METRIC=full_success
  export DRL_MULTI_EARLY_STOP_PATIENCE=2
  export DRL_MULTI_EARLY_STOP_MIN_EPOCHS=4
  export DRL_MULTI_EARLY_STOP_FULL_SUCCESS_DROP=0.15
  export DRL_MULTI_EARLY_STOP_SUCCESS_DROP=0.10
  export DRL_MULTI_EARLY_STOP_TIMEOUT_INCREASE=0.20
  export DRL_MULTI_EARLY_STOP_TIMEOUT_ABSOLUTE=0.30

  export DRL_MULTI_ACTOR_TRAIN_MODE=full
  export DRL_MULTI_USE_LOCAL_CRITIC=1
  export DRL_MULTI_LOCAL_CRITIC_GEOMETRY_ONLY=0
  export DRL_MULTI_LOCAL_CRITIC_CONTEXT_MODE=ego_motion
  export DRL_MULTI_LOCAL_CRITIC_MAX_AGENTS=10
  export DRL_MULTI_ACTIVE_NEIGHBORS_ONLY=1
  export DRL_MULTI_CRITIC_INTERACTION_FRACTION="${DRL_MULTI_CRITIC_INTERACTION_FRACTION:-0.50}"
  export DRL_MULTI_USE_DYNAMIC_REWARD=0
  export DRL_MULTI_USE_DISTANCE_WEIGHTED_REWARD=0
  export DRL_MULTI_USE_LOCAL_NAVIGATION_REWARD=0
  export DRL_MULTI_USE_WALL_CLEARANCE_REWARD=0
  export DRL_MULTI_USE_ORACLE_INTERACTION_ROLLOUT=0
  export DRL_MULTI_ACTOR_INTERACTION_ONLY=0
  export DRL_MULTI_USE_ACTOR_GRADIENT_GATE=0
  export DRL_MULTI_ACTOR_SAFETY_FOCUSED=0
  export DRL_MULTI_ACTOR_ANCHOR_WEIGHT='$ACTOR_ANCHOR_WEIGHT'
  export DRL_MULTI_ACTOR_ANCHOR_SAFE_ONLY=0
  export DRL_MULTI_ACTOR_ANCHOR_SAFE_DISTANCE=2.0
  export DRL_MULTI_ACTOR_Q_NORMALIZATION_ALPHA='$ACTOR_Q_NORM_ALPHA'
  export DRL_MULTI_ACTOR_GRAD_NORM_CLIP='$ACTOR_GRAD_CLIP'

  export DRL_MULTI_REWARD_MODE=average
  export DRL_MULTI_EXPL_NOISE=0.025
  export DRL_MULTI_EXPL_MIN=0.012
  export DRL_MULTI_EXPL_DECAY_STEPS=500000
  export DRL_MULTI_ACTOR_LR=0.000002
  export DRL_MULTI_CRITIC_LR=0.00002
  export DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS='$ACTOR_UPDATE_DELAY'

  cd '$TD3_DIR'
  python3 -u train_velodyne_td3_multi.py >'$log_file' 2>&1
" >"$runner_log" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "Started current generalist retrain."
echo "PID: $(cat "$PID_FILE")"
echo "Model: $MODEL_NAME"
echo "Warm start actor: $BASE_MODEL"
echo "Resume training: $RESUME_TRAINING"
echo "Local critic: enabled"
echo "Actor update delay steps: $ACTOR_UPDATE_DELAY"
echo "Actor anchor weight: $ACTOR_ANCHOR_WEIGHT"
echo "Actor Q normalization alpha: $ACTOR_Q_NORM_ALPHA"
echo "Actor grad norm clip: $ACTOR_GRAD_CLIP"
echo "Train manifest: $TRAIN_MANIFEST"
echo "Eval manifest: $EVAL_MANIFEST"
echo "Max epochs: $MAX_EPOCHS"
echo "Eval episodes: $EVAL_EPISODES"
echo "Log: $log_file"
