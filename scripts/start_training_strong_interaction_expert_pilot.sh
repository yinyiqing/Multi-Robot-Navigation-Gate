#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
VIEW_DIR="$PROJECT_ROOT/experiments/04_保留专门化/05_论文主线/datasets/fixed_v1/views/strong_interaction_v1"
LOG_DIR="$PROJECT_ROOT/logs"
PID_FILE="$PROJECT_ROOT/.train_strong_interaction_expert_pilot.pid"
BASE_ACTOR="TD3_velodyne_multi_v4_curriculum_stage2_to_5d_geo_critic_from_5a_guarded_best"
MODEL_NAME="interaction_expert_temporal_gru_pilot_s20260723"
LAUNCHFILE="multi_robot_scenario_strong_interaction_pilot_5.launch"
ROS_PORT=12403
GAZEBO_PORT=12503

for path in "$VIEW_DIR/train.json.gz" "$VIEW_DIR/validation.json.gz"; do
  [[ -f "$path" ]] || { echo "Strong-interaction view is missing: $path"; exit 1; }
done
[[ -f "$TD3_DIR/pytorch_models/${BASE_ACTOR}_actor.pth" ]] || {
  echo "Frozen 5D actor is missing: $TD3_DIR/pytorch_models/${BASE_ACTOR}_actor.pth"
  exit 1
}
if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Strong-interaction pilot is already running with PID $old_pid"
    exit 1
  fi
  rm -f "$PID_FILE"
fi
if [[ -e "$TD3_DIR/checkpoints/${MODEL_NAME}_latest.pt" ]]; then
  echo "Pilot checkpoint already exists; archive or remove it before a fresh run."
  exit 1
fi

mkdir -p "$LOG_DIR"
python3 "$PROJECT_ROOT/scripts/generate_multi_robot_launch.py" \
  --num-agents 5 \
  --output "$TD3_DIR/assets/$LAUNCHFILE"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$LOG_DIR/train_strong_interaction_expert_pilot_${timestamp}.log"

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
  export DRL_MULTI_MANIFEST_PATH='$VIEW_DIR/train.json.gz'
  export DRL_MULTI_EVAL_MANIFEST_PATH='$VIEW_DIR/validation.json.gz'
  export DRL_MULTI_MANIFEST_SAMPLING=random
  export DRL_STRONG_BASE_MODEL='$BASE_ACTOR'
  export DRL_STRONG_MODEL_NAME='$MODEL_NAME'
  export DRL_STRONG_LAUNCHFILE='$LAUNCHFILE'
  export DRL_STRONG_SEED=20260723
  export DRL_STRONG_HISTORY_LEN=8
  export DRL_STRONG_MAX_AGENT_SAMPLES=40000
  export DRL_STRONG_EVAL_INTERVAL=20000
  export DRL_STRONG_ACTOR_START_SAMPLES=8000
  cd '$PROJECT_ROOT/catkin_ws'
  source devel_isolated/setup.bash
  cd '$TD3_DIR'
  python3 -u train_strong_interaction_expert.py
" >"$log_file" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "Started strong-interaction expert pilot."
echo "PID: $(cat "$PID_FILE")"
echo "Log: $log_file"
echo "Expected runtime: roughly 3-5 hours (baseline + two 140-scene validations)."
