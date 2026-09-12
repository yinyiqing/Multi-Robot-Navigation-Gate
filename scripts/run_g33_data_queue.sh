#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
G33="$BASE/33_双头联合监督预检"
VIEW="$BASE/datasets/fixed_v1/views/g11_a1_gate_v1"
G11="$BASE/11_可部署在线Gate研究/G11_A1_当前协议时序pilot"
NAVIGATION="$ROOT/TD3/pytorch_models/TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best_actor.pth"
INTERACTION="$ROOT/TD3/pytorch_models/interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016_actor.pth"
DETECTOR="$BASE/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"
PROTOCOL="$G33/local_data/protocol"
COLLECTION="$G33/local_data/counterfactual"
LOG="$G33/logs/data_queue"

mkdir -p "$PROTOCOL" "$COLLECTION" "$LOG"
exec > >(tee -a "$LOG/queue.log") 2>&1
echo "[$(date -Is)] G33 data queue started"

# Anchor selection imports the frozen Actor checkpoints through the project venv.
source "$ROOT/env.python.sh"

if [[ ! -f "$PROTOCOL/train_anchors.json" ]]; then
python3 "$ROOT/scripts/select_g27_pilot_anchors.py" \
  --manifest "$VIEW/train.json.gz" \
  --shard-dir "$G11/local_data/shards/train" \
  --navigation-actor "$NAVIGATION" \
  --interaction-actor "$INTERACTION" \
  --output "$PROTOCOL/train_anchors.json" \
  --minimum-step 4 --maximum-step 40 --per-cell 16 --matching \
  --protocol "G33-quality-collection-train-v1"
else
  echo "Reusing existing train selection"
fi

if [[ ! -f "$PROTOCOL/validation_anchors.json" ]]; then
python3 "$ROOT/scripts/select_g27_pilot_anchors.py" \
  --manifest "$VIEW/validation.json.gz" \
  --shard-dir "$G11/local_data/shards/validation" \
  --navigation-actor "$NAVIGATION" \
  --interaction-actor "$INTERACTION" \
  --output "$PROTOCOL/validation_anchors.json" \
  --minimum-step 4 --maximum-step 40 --per-cell 8 --matching \
  --protocol "G33-quality-collection-validation-v1"
else
  echo "Reusing existing validation selection"
fi

set +u
source /opt/ros/noetic/setup.bash
source "$ROOT/catkin_ws/devel_isolated/setup.bash"
set -u

python3 -u "$ROOT/scripts/run_g27_counterfactual_pilot.py" \
  --profile collect \
  --anchors "$PROTOCOL/train_anchors.json" \
  --manifest "$VIEW/train.json.gz" \
  --navigation-actor "$NAVIGATION" \
  --interaction-actor "$INTERACTION" \
  --detector-checkpoint "$DETECTOR" \
  --output-dir "$COLLECTION/train" \
  --jobs "${G33_JOBS:-4}" \
  --base-ros-port 18510 --base-gazebo-port 19510 \
  --seed 20260912 --resume

python3 -u "$ROOT/scripts/run_g27_counterfactual_pilot.py" \
  --profile collect \
  --anchors "$PROTOCOL/validation_anchors.json" \
  --manifest "$VIEW/validation.json.gz" \
  --navigation-actor "$NAVIGATION" \
  --interaction-actor "$INTERACTION" \
  --detector-checkpoint "$DETECTOR" \
  --output-dir "$COLLECTION/validation" \
  --jobs "${G33_JOBS:-4}" \
  --base-ros-port 18610 --base-gazebo-port 19610 \
  --seed 20260912 --resume

python3 "$ROOT/scripts/audit_g33_counterfactual.py" \
  --train-selection "$PROTOCOL/train_anchors.json" \
  --validation-selection "$PROTOCOL/validation_anchors.json" \
  --train-collection "$COLLECTION/train" \
  --validation-collection "$COLLECTION/validation" \
  --output-dir "$COLLECTION/audit"

echo "[$(date -Is)] G33 data queue completed; no training or closed-loop evaluation was started"
