#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
G27="$BASE/27_反事实Reward增强Gate监督"
VIEW="$BASE/datasets/fixed_v1/views/g11_a1_gate_v1"
NAVIGATION="$ROOT/TD3/pytorch_models/TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best_actor.pth"
INTERACTION="$ROOT/TD3/pytorch_models/interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016_actor.pth"
DETECTOR="$BASE/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"
PROFILE="${1:-}"
LIMIT=()

case "$PROFILE" in
  feature-smoke)
    ANCHORS="$G27/local_data/protocol/p1_train_anchors.json"
    MANIFEST="$VIEW/train.json.gz"
    OUTPUT="$G27/local_data/counterfactual/feature_smoke"
    JOBS=1
    BASE_ROS_PORT=18210
    BASE_GAZEBO_PORT=19210
    LIMIT=(--limit 1)
    ;;
  train)
    ANCHORS="$G27/local_data/protocol/p1_train_anchors.json"
    MANIFEST="$VIEW/train.json.gz"
    OUTPUT="$G27/local_data/counterfactual/train"
    JOBS="${G27_P1_JOBS:-4}"
    BASE_ROS_PORT=18310
    BASE_GAZEBO_PORT=19310
    ;;
  validation)
    ANCHORS="$G27/local_data/protocol/p1_validation_anchors.json"
    MANIFEST="$VIEW/validation.json.gz"
    OUTPUT="$G27/local_data/counterfactual/validation"
    JOBS="${G27_P1_JOBS:-4}"
    BASE_ROS_PORT=18410
    BASE_GAZEBO_PORT=19410
    ;;
  *)
    echo "Usage: $0 <feature-smoke|train|validation>" >&2
    exit 2
    ;;
esac

P0_SUMMARY="$G27/local_data/pilot/p0/summary.json"
[[ -f "$P0_SUMMARY" ]] || { echo "G27 P0 summary is missing" >&2; exit 2; }
python3 - "$P0_SUMMARY" <<'PY'
import json, sys
if not json.load(open(sys.argv[1], encoding="utf-8"))["analysis"]["passed"]:
    raise SystemExit("G27 P0 did not pass")
PY

declare -A EXPECTED_SHA=(
  ["$NAVIGATION"]="fa28855049b67b3ee44c66d55d4f14441fc7c521e5429862c75b152f7d5cacc5"
  ["$INTERACTION"]="6ec1942fcd497ab1cc2a85a5aaec8f524395dc21ff21a442dca243a52e917c0b"
  ["$DETECTOR"]="0b914c0d090bbaba0a2be63c0d75d88580bf4d71778a7f1749c63989e3dbbd56"
)
for path in "${!EXPECTED_SHA[@]}"; do
  [[ -f "$path" ]] || { echo "Required frozen input is missing: $path" >&2; exit 1; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "${EXPECTED_SHA[$path]}" ]] || {
    echo "Frozen input hash mismatch: $path" >&2
    exit 1
  }
done
for path in "$ANCHORS" "$MANIFEST"; do
  [[ -f "$path" ]] || { echo "Required protocol input is missing: $path" >&2; exit 1; }
done

RESUME=()
if [[ -d "$OUTPUT" ]] && find "$OUTPUT" -mindepth 1 -print -quit | grep -q .; then
  RESUME=(--resume)
fi

set +u
source /opt/ros/noetic/setup.bash
source "$ROOT/env.python.sh"
source "$ROOT/catkin_ws/devel_isolated/setup.bash"
set -u

python3 -u "$ROOT/scripts/run_g27_counterfactual_pilot.py" \
  --profile collect \
  --anchors "$ANCHORS" \
  --manifest "$MANIFEST" \
  --navigation-actor "$NAVIGATION" \
  --interaction-actor "$INTERACTION" \
  --detector-checkpoint "$DETECTOR" \
  --output-dir "$OUTPUT" \
  --jobs "$JOBS" \
  --base-ros-port "$BASE_ROS_PORT" \
  --base-gazebo-port "$BASE_GAZEBO_PORT" \
  "${RESUME[@]}" \
  "${LIMIT[@]}"
