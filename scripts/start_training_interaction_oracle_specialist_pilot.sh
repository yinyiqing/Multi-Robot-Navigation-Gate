#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
VIEW_DIR="$PROJECT_ROOT/experiments/04_保留专门化/05_论文主线/datasets/fixed_v1/views/strong_interaction_curriculum_v1"
TRAIN_MANIFEST="${DRL_MULTI_TRAIN_MANIFEST:-$VIEW_DIR/stage2_train.json.gz}"
EVAL_MANIFEST="${DRL_MULTI_EVAL_MANIFEST:-$VIEW_DIR/validation.json.gz}"
LOG_DIR="$PROJECT_ROOT/logs"
BASE_MODEL="${DRL_MULTI_BASE_MODEL:-TD3_velodyne_multi_v4_curriculum_stage2_to_5d_geo_critic_from_5a_guarded_best}"
ORACLE_WEAK_ACTOR_NAME="${DRL_MULTI_ORACLE_WEAK_ACTOR_NAME:-$BASE_MODEL}"
LOAD_ACTOR_ONLY="${DRL_MULTI_LOAD_ACTOR_ONLY:-0}"
MODEL_NAME="${DRL_MULTI_TRAIN_FILE_NAME:-interaction_oracle_specialist_pilot_s20260724}"
SAFE_MODEL="${MODEL_NAME//[^A-Za-z0-9_]/_}"
PID_FILE="${DRL_MULTI_PID_FILE:-$PROJECT_ROOT/.train_${SAFE_MODEL}.pid}"
LAUNCHFILE="multi_robot_scenario_strong_interaction_pilot_5.launch"
ROS_PORT="${DRL_MULTI_ROS_PORT:-13003}"
GAZEBO_PORT="${DRL_MULTI_GAZEBO_PORT:-13103}"
ROBOT_SAFE_DISTANCE="${DRL_MULTI_ROBOT_SAFE_DISTANCE:-1.2}"
ROBOT_PROXIMITY_PENALTY_WEIGHT="${DRL_MULTI_ROBOT_PROXIMITY_PENALTY_WEIGHT:-5.0}"
ROBOT_PROXIMITY_SPEED_PENALTY_WEIGHT="${DRL_MULTI_ROBOT_PROXIMITY_SPEED_PENALTY_WEIGHT:-10.0}"
ROBOT_CLEARANCE_REWARD_WEIGHT="${DRL_MULTI_ROBOT_CLEARANCE_REWARD_WEIGHT:-20.0}"
ROBOT_CLEARANCE_REWARD_MAX_GAIN="${DRL_MULTI_ROBOT_CLEARANCE_REWARD_MAX_GAIN:-0.1}"
RESUME_TRAINING="${DRL_MULTI_RESUME_TRAINING:-0}"
TRAIN_SEED="${DRL_MULTI_SEED:-20260724}"
MAX_EPOCHS="${DRL_MULTI_MAX_EPOCHS:-2}"

[[ "$LOAD_ACTOR_ONLY" == 0 || "$LOAD_ACTOR_ONLY" == 1 ]] || {
  echo "DRL_MULTI_LOAD_ACTOR_ONLY must be 0 or 1."
  exit 2
}
[[ "$RESUME_TRAINING" == 0 || "$RESUME_TRAINING" == 1 ]] || {
  echo "DRL_MULTI_RESUME_TRAINING must be 0 or 1."
  exit 2
}
[[ "$TRAIN_SEED" =~ ^[0-9]+$ ]] || {
  echo "DRL_MULTI_SEED must be a non-negative integer."
  exit 2
}
[[ "$MAX_EPOCHS" =~ ^[1-9][0-9]*$ ]] || {
  echo "DRL_MULTI_MAX_EPOCHS must be a positive integer."
  exit 2
}

for path in "$TRAIN_MANIFEST" "$EVAL_MANIFEST"; do
  [[ -f "$path" ]] || { echo "Fixed five-agent interaction split is missing: $path"; exit 1; }
done
required_suffixes=(actor)
if [[ "$LOAD_ACTOR_ONLY" == 0 ]]; then
  required_suffixes+=(critic)
fi
for suffix in "${required_suffixes[@]}"; do
  [[ -f "$TD3_DIR/pytorch_models/${BASE_MODEL}_${suffix}.pth" ]] || {
    echo "Warm-start ${suffix} is missing: $TD3_DIR/pytorch_models/${BASE_MODEL}_${suffix}.pth"
    exit 1
  }
done
[[ -f "$TD3_DIR/pytorch_models/${ORACLE_WEAK_ACTOR_NAME}_actor.pth" ]] || {
  echo "Oracle weak Actor is missing: $TD3_DIR/pytorch_models/${ORACLE_WEAK_ACTOR_NAME}_actor.pth"
  exit 1
}
[[ -f "$TD3_DIR/assets/$LAUNCHFILE" ]] || {
  echo "Five-agent launch file is missing: $TD3_DIR/assets/$LAUNCHFILE"
  exit 1
}

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Oracle-specialist pilot is already running with PID $old_pid"
    exit 1
  fi
  unlink "$PID_FILE"
fi
if [[ "$RESUME_TRAINING" == 0 && -e "$TD3_DIR/checkpoints/${MODEL_NAME}_latest.pt" ]]; then
  echo "Pilot checkpoint already exists; archive it before starting a fresh run."
  exit 1
fi
if [[ "$RESUME_TRAINING" == 1 && ! -e "$TD3_DIR/checkpoints/${MODEL_NAME}_latest.pt" ]]; then
  echo "Cannot resume because checkpoint is missing: $TD3_DIR/checkpoints/${MODEL_NAME}_latest.pt"
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
log_file="$LOG_DIR/train_${SAFE_MODEL}_${timestamp}.log"

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
  export DRL_MULTI_NUM_AGENTS=5
  export DRL_MULTI_SEED='$TRAIN_SEED'
  export DRL_MULTI_TRAIN_LAUNCHFILE='$LAUNCHFILE'
  export DRL_MULTI_SCENARIO=manifest
  export DRL_MULTI_MANIFEST_PATH='$TRAIN_MANIFEST'
  export DRL_MULTI_EVAL_MANIFEST_PATH='$EVAL_MANIFEST'
  export DRL_MULTI_MANIFEST_SAMPLING=random
  export DRL_MULTI_TRAIN_FILE_NAME='$MODEL_NAME'
  export DRL_MULTI_LOAD_MODEL=1
  export DRL_MULTI_LOAD_ACTOR_ONLY='$LOAD_ACTOR_ONLY'
  export DRL_MULTI_LOAD_MODEL_NAME='$BASE_MODEL'
  export DRL_MULTI_RESUME_TRAINING='$RESUME_TRAINING'
  export DRL_MULTI_MAX_EPOCHS='$MAX_EPOCHS'
  export DRL_MULTI_EVAL_EPISODES=140
  export DRL_MULTI_EVAL_FREQ_AGENT_SAMPLES=20000
  export DRL_MULTI_BEST_METRIC=full_success
  export DRL_MULTI_TRAINING_VERSION='interaction-oracle-specialist-pilot-v6-focused-actor'
  export DRL_MULTI_ACTOR_TRAIN_MODE=full
  export DRL_MULTI_USE_ORACLE_INTERACTION_ROLLOUT=1
  export DRL_MULTI_ORACLE_WEAK_ACTOR_NAME='$ORACLE_WEAK_ACTOR_NAME'
  export DRL_MULTI_ORACLE_INTERACTION_DISTANCE=2.0
  export DRL_MULTI_ROBOT_SAFE_DISTANCE='$ROBOT_SAFE_DISTANCE'
  export DRL_MULTI_ROBOT_PROXIMITY_PENALTY_WEIGHT='$ROBOT_PROXIMITY_PENALTY_WEIGHT'
  export DRL_MULTI_ROBOT_PROXIMITY_SPEED_PENALTY_WEIGHT='$ROBOT_PROXIMITY_SPEED_PENALTY_WEIGHT'
  export DRL_MULTI_ROBOT_CLEARANCE_REWARD_WEIGHT='$ROBOT_CLEARANCE_REWARD_WEIGHT'
  export DRL_MULTI_ROBOT_CLEARANCE_REWARD_MAX_GAIN='$ROBOT_CLEARANCE_REWARD_MAX_GAIN'
  export DRL_MULTI_ACTOR_INTERACTION_ONLY=1
  export DRL_MULTI_USE_DYNAMIC_REWARD=1
  export DRL_MULTI_REWARD_MODE=average
  export DRL_MULTI_REWARD_SELF_WEIGHT=0.8
  export DRL_MULTI_USE_DISTANCE_WEIGHTED_REWARD=1
  export DRL_MULTI_REWARD_SIGMA=2.0
  export DRL_MULTI_USE_LOCAL_CRITIC=1
  export DRL_MULTI_LOCAL_CRITIC_GEOMETRY_ONLY=0
  export DRL_MULTI_LOCAL_CRITIC_CONTEXT_MODE=ego_motion
  export DRL_MULTI_LOCAL_CRITIC_MAX_AGENTS=10
  export DRL_MULTI_ACTIVE_NEIGHBORS_ONLY=1
  export DRL_MULTI_FORWARD_REWARD_WEIGHT=0.0
  export DRL_MULTI_STAGNATION_PENALTY_WEIGHT=0.0
  export DRL_MULTI_USE_ANTI_STAGNATION_REWARD=0
  export DRL_MULTI_USE_LOCAL_NAVIGATION_REWARD=0
  export DRL_MULTI_USE_WALL_CLEARANCE_REWARD=0
  export DRL_MULTI_EXPL_NOISE=0.08
  export DRL_MULTI_CRITIC_WARMUP_EXPL_NOISE=0.30
  export DRL_MULTI_EXPL_MIN=0.03
  export DRL_MULTI_EXPL_DECAY_STEPS=80000
  export DRL_MULTI_ACTOR_LR=0.000001
  export DRL_MULTI_CRITIC_LR=0.00008
  export DRL_MULTI_CRITIC_INTERACTION_FRACTION=0.75
  export DRL_MULTI_USE_ACTOR_GRADIENT_GATE=1
  export DRL_MULTI_ACTOR_GRADIENT_SAFETY_DISTANCE=1.2
  export DRL_MULTI_ACTOR_GRADIENT_GATE_BATCH_SIZE=512
  export DRL_MULTI_ACTOR_GRADIENT_GATE_MIN_SAMPLES=32
  export DRL_MULTI_ACTOR_GRADIENT_MAX_LINEAR_POSITIVE_SHARE=0.9
  export DRL_MULTI_ACTOR_GRADIENT_MAX_ANGULAR_ONE_SIDED_SHARE=0.9
  export DRL_MULTI_CRITIC_SAFETY_RANKING_WEIGHT=5.0
  export DRL_MULTI_CRITIC_SAFETY_RANKING_DISTANCE=1.0
  export DRL_MULTI_CRITIC_SAFETY_RANKING_MIN_CLOSING_SPEED=0.1
  export DRL_MULTI_CRITIC_SAFETY_RANKING_LINEAR_DELTA=0.4
  export DRL_MULTI_CRITIC_SAFETY_RANKING_MARGIN=0.1
  export DRL_MULTI_ACTOR_SAFETY_FOCUSED=1
  export DRL_MULTI_ACTOR_SAFETY_CANDIDATE_BATCH_SIZE=256
  export DRL_MULTI_ACTOR_SAFETY_MIN_SAMPLES=16
  export DRL_MULTI_ACTOR_SAFETY_DISTANCE=1.0
  export DRL_MULTI_ACTOR_SAFETY_MIN_CLOSING_SPEED=0.1
  export DRL_MULTI_ACTOR_ANGULAR_ANCHOR_WEIGHT=2.0
  export DRL_MULTI_ACTOR_ANCHOR_WEIGHT=0.0
  export DRL_MULTI_ACTOR_Q_NORMALIZATION_ALPHA=0.0
  export DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS=21000
  export DRL_MULTI_POLICY_FREQ=2
  cd '$TD3_DIR'
  python3 -u train_velodyne_td3_multi.py
" >"$log_file" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "Started five-agent oracle-specialist pilot."
echo "PID: $(cat "$PID_FILE")"
echo "Model: $MODEL_NAME"
echo "Warm start: $BASE_MODEL"
if [[ "$LOAD_ACTOR_ONLY" == 1 ]]; then
  echo "Warm start mode: actor-only"
else
  echo "Warm start mode: actor-and-critic"
fi
echo "Train manifest: $TRAIN_MANIFEST"
echo "Validation manifest: $EVAL_MANIFEST"
echo "Oracle: trainable Actor at <=2.0 m; frozen $ORACLE_WEAK_ACTOR_NAME otherwise"
echo "Actor updates: interaction transitions only"
echo "Robot safe distance reward: $ROBOT_SAFE_DISTANCE m"
echo "Robot proximity penalty weight: $ROBOT_PROXIMITY_PENALTY_WEIGHT"
echo "Robot proximity speed penalty weight: $ROBOT_PROXIMITY_SPEED_PENALTY_WEIGHT"
echo "Robot clearance reward weight: $ROBOT_CLEARANCE_REWARD_WEIGHT"
echo "Robot clearance reward max gain: $ROBOT_CLEARANCE_REWARD_MAX_GAIN m/step"
echo "Resume training: $RESUME_TRAINING"
echo "Seed: $TRAIN_SEED"
echo "Maximum epochs: $MAX_EPOCHS"
echo "Epoch 1: frozen Actor baseline; Epoch 2 onward: interaction-only Actor training"
echo "Log: $log_file"
echo "Expected runtime: depends on manifest size; the full 2560-scene pool is roughly 3-4 hours per epoch."
