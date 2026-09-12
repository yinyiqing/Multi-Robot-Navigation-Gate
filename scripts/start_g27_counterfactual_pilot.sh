#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
G27="$BASE/27_反事实Reward增强Gate监督"
ANCHORS="$G27/local_data/protocol/p0_anchors.json"
MANIFEST="$BASE/datasets/fixed_v1/views/g11_a1_gate_v1/train.json.gz"
NAVIGATION="$ROOT/TD3/pytorch_models/TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best_actor.pth"
INTERACTION="$ROOT/TD3/pytorch_models/interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016_actor.pth"
DETECTOR="$BASE/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"
PROFILE="${1:-}"
RESUME=()

case "$PROFILE" in
  smoke)
    OUTPUT="$G27/local_data/pilot/smoke"
    JOBS=1
    LIMIT=(--limit 1)
    ;;
  pilot)
    OUTPUT="$G27/local_data/pilot/p0"
    JOBS="${G27_JOBS:-2}"
    LIMIT=()
    SMOKE_SUMMARY="$G27/local_data/pilot/smoke/summary.json"
    [[ -f "$SMOKE_SUMMARY" ]] || {
      echo "G27 smoke summary is missing; run smoke first" >&2
      exit 2
    }
    python3 - "$SMOKE_SUMMARY" <<'PY'
import json, sys
if not json.load(open(sys.argv[1], encoding="utf-8"))["analysis"]["passed"]:
    raise SystemExit("G27 smoke did not pass")
PY
    ;;
  *)
    echo "Usage: $0 <smoke|pilot>" >&2
    exit 2
    ;;
esac

if [[ -d "$OUTPUT" ]] && find "$OUTPUT" -mindepth 1 -print -quit | grep -q .; then
  RESUME=(--resume)
fi

declare -A EXPECTED_SHA=(
  ["$MANIFEST"]="a97534ed22d1b4b12951cd42b80d515037774830f38cb432926c0e9f50379026"
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

if [[ ! -f "$ANCHORS" ]]; then
  python3 "$ROOT/scripts/select_g27_pilot_anchors.py"
fi

set +u
source /opt/ros/noetic/setup.bash
source "$ROOT/env.python.sh"
source "$ROOT/catkin_ws/devel_isolated/setup.bash"
set -u

python3 -u "$ROOT/scripts/run_g27_counterfactual_pilot.py" \
  --profile "$PROFILE" \
  --anchors "$ANCHORS" \
  --manifest "$MANIFEST" \
  --navigation-actor "$NAVIGATION" \
  --interaction-actor "$INTERACTION" \
  --detector-checkpoint "$DETECTOR" \
  --output-dir "$OUTPUT" \
  --jobs "$JOBS" \
  "${RESUME[@]}" \
  "${LIMIT[@]}"
