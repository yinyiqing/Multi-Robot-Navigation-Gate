#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
VIEW_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/robot_perception_v1"
OUTPUT_ROOT="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/results/06_Gate开发/D5_G2_interaction_gate_v1/local_data/counterfactual"
DETECTOR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"
GENERALIST="$TD3_DIR/pytorch_models/TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best_actor.pth"
STRONG="$TD3_DIR/pytorch_models/interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016_actor.pth"
PROFILE="${1:-}"

case "$PROFILE" in
  pilot-train)
    SPLIT=train
    MANIFEST="$VIEW_DIR/pilot_train.json.gz"
    OUTPUT_DIR="$OUTPUT_ROOT/pilot_train"
    ROS_PORT=12833
    GAZEBO_PORT=12933
    ;;
  pilot-validation)
    SPLIT=validation
    MANIFEST="$VIEW_DIR/pilot_validation.json.gz"
    OUTPUT_DIR="$OUTPUT_ROOT/pilot_validation"
    ROS_PORT=13033
    GAZEBO_PORT=13133
    ;;
  *)
    echo "Usage: $0 <pilot-train|pilot-validation>" >&2
    exit 2
    ;;
esac

for required in "$MANIFEST" "$DETECTOR" "$GENERALIST" "$STRONG"; do
  [[ -f "$required" ]] || { echo "Required input is missing: $required" >&2; exit 1; }
done

PID_FILE="$PROJECT_ROOT/.actor_counterfactual_${PROFILE//-/_}.pid"
if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Actor counterfactual $PROFILE is already running with PID $old_pid" >&2
    exit 1
  fi
  unlink "$PID_FILE"
fi

target_episodes="${DRL_G2B_TARGET_EPISODES:-1}"
if [[ "$target_episodes" != "1" ]]; then
  echo "G2-B v1 failed repeatability; only the 1-episode audit is allowed." >&2
  exit 2
fi
horizon="${DRL_G2B_HORIZON:-8}"
anchor_stride="${DRL_G2B_ANCHOR_STRIDE:-1}"
max_anchors="${DRL_G2B_MAX_ANCHORS_PER_EPISODE:-4}"
agents_per_anchor="${DRL_G2B_AGENTS_PER_ANCHOR:-2}"
max_episode_steps="${DRL_G2B_MAX_EPISODE_STEPS:-300}"
mkdir -p "$PROJECT_ROOT/logs" "$OUTPUT_DIR"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$PROJECT_ROOT/logs/collect_actor_counterfactual_${PROFILE//-/_}_${timestamp}.log"

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
  cd '$TD3_DIR'
  python3 -u collect_actor_counterfactuals.py \
    --manifest '$MANIFEST' \
    --output-dir '$OUTPUT_DIR' \
    --generalist-actor '$GENERALIST' \
    --strong-actor '$STRONG' \
    --detector-checkpoint '$DETECTOR' \
    --episodes '$target_episodes' \
    --horizon '$horizon' \
    --anchor-stride '$anchor_stride' \
    --max-anchors-per-episode '$max_anchors' \
    --agents-per-anchor '$agents_per_anchor' \
    --max-episode-steps '$max_episode_steps' \
    --split '$SPLIT' \
    --repeat-baseline
" >"$log_file" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "Started actor counterfactual $PROFILE collection."
echo "PID: $(cat "$PID_FILE")"
echo "Episodes: $target_episodes"
echo "Horizon: $horizon"
echo "Log: $log_file"
echo "Output: $OUTPUT_DIR"
