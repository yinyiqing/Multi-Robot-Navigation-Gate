#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATE_ROOT="$ROOT/experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究"
A1_DIR="$GATE_ROOT/G11_A1_当前协议时序pilot"
RUN_DIR="$GATE_ROOT/G11_F_epoch17_gate_v1"
VIEW_DIR="$ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/g11_a1_gate_v1"
INTERACTION_ACTOR="$ROOT/TD3/pytorch_models/avoidance_actor_from_5a_balanced_continue_e20_s20260813_best_actor.pth"
OUTPUT_DIR="$RUN_DIR/local_data/a1_training/seed20260804"
LOG_DIR="$ROOT/logs/active/g11_f_epoch17_gate"
LOCK_FILE="$RUN_DIR/local_data/.a1_training.lock"

mkdir -p "$(dirname "$LOCK_FILE")"
exec 9>"$LOCK_FILE"
flock -n 9 || { echo "G11-F epoch-17 A1 training is already running" >&2; exit 1; }
[[ ! -f "$OUTPUT_DIR/summary.json" ]] || {
  echo "G11-F epoch-17 A1 training already completed" >&2
  exit 1
}

[[ "$(sha256sum "$INTERACTION_ACTOR" | awk '{print $1}')" == \
  "149c2e42848ecc9bc478cbed7fd89b9062936dbd5c669b55e6964441685155a5" ]] || {
  echo "epoch-17 Actor SHA-256 mismatch" >&2
  exit 1
}
/usr/bin/python3 "$ROOT/scripts/audit_g11_a1_shards.py" >/dev/null

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
LOG_FILE="$LOG_DIR/a1_seed20260804_$(date +%Y%m%d_%H%M%S).log"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

cd "$ROOT"
/usr/bin/time -f 'elapsed=%E maxrss_kb=%M' nice -n 15 /usr/bin/python3 \
  scripts/train_temporal_interaction_gate.py \
  --experiment-id G11-F-A1 \
  --train-dir "$A1_DIR/local_data/shards/train" \
  --validation-dir "$A1_DIR/local_data/shards/validation" \
  --train-manifest "$VIEW_DIR/train.json.gz" \
  --validation-manifest "$VIEW_DIR/validation.json.gz" \
  --interaction-actor "$INTERACTION_ACTOR" \
  --expected-interaction-actor-sha256 \
    149c2e42848ecc9bc478cbed7fd89b9062936dbd5c669b55e6964441685155a5 \
  --labels any \
  --models S0 T1 \
  --threshold-policy match-s0-fpr \
  --epochs 40 \
  --seed 20260804 \
  --device cpu \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_FILE"

echo "G11-F epoch-17 A1 log: $LOG_FILE"
