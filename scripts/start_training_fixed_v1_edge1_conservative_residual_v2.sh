#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
VIEW_DIR="$PROJECT_ROOT/experiments/04_保留专门化/05_论文主线/datasets/fixed_v1/views/edge1_pilot"
ARCHIVE_DIR="$PROJECT_ROOT/experiments/04_保留专门化/05_论文主线/results/D4_interaction_edge1_residual_pilot_s20260720"
LOG_DIR="$PROJECT_ROOT/logs"
PID_FILE="$PROJECT_ROOT/.train_fixed_v1_edge1_conservative_residual_v2.pid"
INIT_MODEL="interaction_expert_edge1_residual_epoch1_init"
MODEL_NAME="interaction_expert_edge1_conservative_residual_v2_s20260720"
NUM_AGENTS=5
SEED=20260720
ROS_PORT=12601
GAZEBO_PORT=12701
EVAL_EPISODES=423
EVAL_FREQ_AGENT_SAMPLES=40000
LAUNCHFILE="multi_robot_scenario_fixed_v1_edge1_conservative_residual_v2_${NUM_AGENTS}.launch"

for path in \
  "$VIEW_DIR/train.json.gz" \
  "$VIEW_DIR/validation.json.gz" \
  "$ARCHIVE_DIR/epoch001_best_actor.pth" \
  "$ARCHIVE_DIR/epoch001_best_critic.pth"; do
  if [[ ! -f "$path" ]]; then
    echo "Required pilot artifact is missing: $path"
    exit 1
  fi
done
if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Edge-1 conservative residual v2 is already running with PID $old_pid"
    exit 1
  fi
fi
for suffix in latest.pt best.pt; do
  if [[ -e "$TD3_DIR/checkpoints/${MODEL_NAME}_${suffix}" ]]; then
    echo "V2 checkpoint already exists: $TD3_DIR/checkpoints/${MODEL_NAME}_${suffix}"
    exit 1
  fi
done

mkdir -p "$LOG_DIR" "$TD3_DIR/pytorch_models"
cp "$ARCHIVE_DIR/epoch001_best_actor.pth" "$TD3_DIR/pytorch_models/${INIT_MODEL}_actor.pth"
cp "$ARCHIVE_DIR/epoch001_best_critic.pth" "$TD3_DIR/pytorch_models/${INIT_MODEL}_critic.pth"
python3 "$PROJECT_ROOT/scripts/generate_multi_robot_launch.py" \
  --num-agents "$NUM_AGENTS" \
  --output "$TD3_DIR/assets/$LAUNCHFILE"

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$LOG_DIR/train_fixed_v1_edge1_conservative_residual_v2_${timestamp}.log"

setsid bash -lc "
  set -eo pipefail
  cleanup() {
    pgid=\"\$(ps -o pgid= -p \$\$ | tr -d ' ')\"
    ps -eo pid=,pgid= | awk -v pgid=\"\$pgid\" -v self=\"\$\$\" \\
      '\$2 == pgid && \$1 != self { print \$1 }' | xargs -r kill 2>/dev/null || true
    rm -f '$PID_FILE'
  }
  trap cleanup EXIT
  source /opt/ros/noetic/setup.bash
  source '$PROJECT_ROOT/env.python.sh'
  export ROS_HOSTNAME=localhost
  export ROS_MASTER_URI=http://localhost:$ROS_PORT
  export ROS_PORT_SIM=$ROS_PORT
  export GAZEBO_MASTER_URI=http://localhost:$GAZEBO_PORT
  export GAZEBO_RESOURCE_PATH='$PROJECT_ROOT/catkin_ws/src/multi_robot_scenario/launch'
  export DRL_MULTI_NUM_AGENTS=$NUM_AGENTS
  export DRL_MULTI_SEED=$SEED
  export DRL_MULTI_TRAIN_LAUNCHFILE='$LAUNCHFILE'
  export DRL_MULTI_SCENARIO=manifest
  export DRL_MULTI_MANIFEST_PATH='$VIEW_DIR/train.json.gz'
  export DRL_MULTI_EVAL_MANIFEST_PATH='$VIEW_DIR/validation.json.gz'
  export DRL_MULTI_MANIFEST_SAMPLING=cycle
  export DRL_MULTI_TRAIN_FILE_NAME='$MODEL_NAME'
  export DRL_MULTI_LOAD_MODEL=1
  export DRL_MULTI_LOAD_ACTOR_ONLY=0
  export DRL_MULTI_LOAD_MODEL_NAME='$INIT_MODEL'
  export DRL_MULTI_RESUME_TRAINING=0
  export DRL_MULTI_MAX_EPOCHS=1
  export DRL_MULTI_EVAL_EPISODES=$EVAL_EPISODES
  export DRL_MULTI_EVAL_FREQ_AGENT_SAMPLES=$EVAL_FREQ_AGENT_SAMPLES
  export DRL_MULTI_BEST_METRIC=full_success
  export DRL_MULTI_TRAINING_VERSION='interaction-edge1-conservative-residual-v2'
  export DRL_MULTI_ACTOR_TRAIN_MODE=residual
  export DRL_MULTI_RESIDUAL_HIDDEN_DIM=128
  export DRL_MULTI_RESIDUAL_SCALE=0.10
  export DRL_MULTI_USE_DYNAMIC_REWARD=1
  export DRL_MULTI_REWARD_MODE=average
  export DRL_MULTI_REWARD_SELF_WEIGHT=0.8
  export DRL_MULTI_USE_DISTANCE_WEIGHTED_REWARD=1
  export DRL_MULTI_REWARD_SIGMA=2.0
  export DRL_MULTI_USE_LOCAL_CRITIC=1
  export DRL_MULTI_LOCAL_CRITIC_GEOMETRY_ONLY=1
  export DRL_MULTI_LOCAL_CRITIC_MAX_AGENTS=10
  export DRL_MULTI_ACTIVE_NEIGHBORS_ONLY=1
  export DRL_MULTI_USE_LOCAL_NAVIGATION_REWARD=0
  export DRL_MULTI_USE_WALL_CLEARANCE_REWARD=0
  export DRL_MULTI_EXPL_NOISE=0.03
  export DRL_MULTI_EXPL_MIN=0.015
  export DRL_MULTI_EXPL_DECAY_STEPS=80000
  export DRL_MULTI_ACTOR_LR=0.0001
  export DRL_MULTI_CRITIC_LR=0.00008
  export DRL_MULTI_ACTOR_ANCHOR_WEIGHT=2.5
  export DRL_MULTI_ACTOR_Q_NORMALIZATION_ALPHA=1.0
  export DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS=0
  export DRL_MULTI_POLICY_FREQ=2
  cd '$PROJECT_ROOT/catkin_ws'
  source devel_isolated/setup.bash
  cd '$TD3_DIR'
  python3 -u train_velodyne_td3_multi.py
" >"$log_file" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "Started fixed-v1 edge-1 conservative residual v2."
echo "PID: $(cat "$PID_FILE")"
echo "Initialization: archived epoch-1 residual Actor/Critic"
echo "Actor Q normalization alpha: 1.0"
echo "Base-action anchor weight: 2.5"
echo "Validation after: $EVAL_FREQ_AGENT_SAMPLES agent samples"
echo "Log: $log_file"
