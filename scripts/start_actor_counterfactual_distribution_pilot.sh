#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
G4_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/07_冲突拓扑组合泛化/G4_G2B_v2多次Rollout标签"
MANIFEST="$G4_DIR/pilot_manifest.json"
DETECTOR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"
GENERALIST="$TD3_DIR/pytorch_models/TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best_actor.pth"
STRONG="$TD3_DIR/pytorch_models/interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016_actor.pth"
PROFILE="${1:-}"

case "$PROFILE" in
  smoke)
    EPISODES=1
    HORIZON=2
    MAX_ANCHORS=1
    ROLLOUTS=2
    BATCHES=2
    BOOTSTRAP_RESAMPLES=500
    ROS_PORT=14633
    GAZEBO_PORT=14733
    ;;
  pilot)
    echo "G4 smoke failed anchor repeatability; the formal pilot is disabled." >&2
    exit 2
    ;;
  *)
    echo "Usage: $0 <smoke|pilot>" >&2
    exit 2
    ;;
esac

for required in "$MANIFEST" "$DETECTOR" "$GENERALIST" "$STRONG"; do
  [[ -f "$required" ]] || { echo "Required input is missing: $required" >&2; exit 1; }
done

PID_FILE="$PROJECT_ROOT/.g4_counterfactual_distribution_${PROFILE}.pid"
if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "G4 counterfactual distribution $PROFILE is already running with PID $old_pid" >&2
    exit 1
  fi
  unlink "$PID_FILE"
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$G4_DIR/local_data/${PROFILE}_${timestamp}"
OUTPUT_DIR="$RUN_DIR/shards"
LOG_FILE="$RUN_DIR/collector.log"
mkdir -p "$OUTPUT_DIR"

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
    --episodes '$EPISODES' \
    --horizon '$HORIZON' \
    --anchor-start-step 4 \
    --anchor-stride 4 \
    --max-anchors-per-episode '$MAX_ANCHORS' \
    --agents-per-anchor 1 \
    --rollouts-per-actor '$ROLLOUTS' \
    --label-batches '$BATCHES' \
    --bootstrap-resamples '$BOOTSTRAP_RESAMPLES' \
    --max-episode-steps 40 \
    --seed 20260802 \
    --split validation \
    --device cpu
" >"$LOG_FILE" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "Started G4 counterfactual distribution $PROFILE."
echo "PID: $(cat "$PID_FILE")"
echo "Episodes: $EPISODES"
echo "Rollouts per actor/batches: $ROLLOUTS/$BATCHES"
echo "Horizon/anchors: $HORIZON/$MAX_ANCHORS"
echo "Log: $LOG_FILE"
echo "Output: $OUTPUT_DIR"
