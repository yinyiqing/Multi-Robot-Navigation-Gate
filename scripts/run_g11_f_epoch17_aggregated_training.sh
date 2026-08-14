#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATE_ROOT="$ROOT/experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究"
OLD_A1="$GATE_ROOT/G11_A1_当前协议时序pilot"
RUN_DIR="$GATE_ROOT/G11_F_epoch17_gate_v1"
VIEW_DIR="$ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/g11_a1_gate_v1"
DETECTOR="$ROOT/experiments/03_保留专门化/02_论文主线/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"
STANDARD_ACTOR="$ROOT/TD3/pytorch_models/TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best_actor.pth"
INTERACTION_ACTOR="$ROOT/TD3/pytorch_models/avoidance_actor_from_5a_balanced_continue_e20_s20260813_best_actor.pth"
A1_MAIN="$RUN_DIR/local_data/a1_training/seed20260804"
OUTPUT_DIR="$RUN_DIR/local_data/aggregated_training/seed20260804"
LOG_DIR="$ROOT/logs/active/g11_f_epoch17_gate"
LOCK_FILE="$RUN_DIR/local_data/.aggregated_training.lock"

mkdir -p "$(dirname "$LOCK_FILE")"
exec 9>"$LOCK_FILE"
flock -n 9 || { echo "G11-F aggregated Gate training is already running" >&2; exit 1; }
[[ ! -f "$OUTPUT_DIR/summary.json" ]] || {
  echo "G11-F aggregated Gate training already completed" >&2; exit 1
}

/usr/bin/python3 "$ROOT/scripts/audit_g11_a1_shards.py" >/dev/null
/usr/bin/python3 "$ROOT/scripts/audit_g11_b_student_shards.py" \
  --profile train \
  --route "$RUN_DIR" \
  --manifest "$VIEW_DIR/train.json.gz" \
  --run-metadata "$RUN_DIR/student_run_metadata.json" >/dev/null

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
LOG_FILE="$LOG_DIR/aggregated_seed20260804_$(date +%Y%m%d_%H%M%S).log"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

cd "$ROOT"
/usr/bin/time -f 'elapsed=%E maxrss_kb=%M' nice -n 15 /usr/bin/python3 \
  scripts/train_g11_b_aggregated_gate.py \
  --experiment-id G11-F-B2 \
  --a1-train-dir "$OLD_A1/local_data/shards/train" \
  --student-train-dir "$RUN_DIR/local_data/student_shards/train" \
  --validation-dir "$OLD_A1/local_data/shards/validation" \
  --train-manifest "$VIEW_DIR/train.json.gz" \
  --validation-manifest "$VIEW_DIR/validation.json.gz" \
  --detector-checkpoint "$DETECTOR" \
  --standard-actor "$STANDARD_ACTOR" \
  --interaction-actor "$INTERACTION_ACTOR" \
  --expected-interaction-actor-sha256 \
    149c2e42848ecc9bc478cbed7fd89b9062936dbd5c669b55e6964441685155a5 \
  --a1-main-summary "$A1_MAIN/summary.json" \
  --a1-main-checkpoint "$A1_MAIN/any/T1/best.pt" \
  --student-audit "$RUN_DIR/local_data/train_audit.json" \
  --expected-a1-summary-sha256 \
    0c2391bd736806feb814edf8e4b638f53114dd96d570cc5c5a79265f8ff00ff4 \
  --expected-a1-checkpoint-sha256 \
    b28e81d341c145d6fa8c881dd98c7ece5285231e7d080b3f71afcd2dfe3a0beb \
  --expected-student-dataset-sha256 \
    5037144924ceb5e433a5e02a17cdffa5a4338f016f08208dc7a64854548887e8 \
  --epochs 40 \
  --seed 20260804 \
  --device cpu \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_FILE"

echo "G11-F aggregated Gate log: $LOG_FILE"
