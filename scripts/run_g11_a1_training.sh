#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究/G11_A1_当前协议时序pilot"
VIEW_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/g11_a1_gate_v1"
TRAIN_DIR="$RUN_DIR/local_data/shards/train"
VALIDATION_DIR="$RUN_DIR/local_data/shards/validation"
SEED="${1:-20260804}"

[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "Seed must be an integer" >&2; exit 2; }
for profile in train validation; do
  pid_file="$PROJECT_ROOT/.robot_perception_g11_a1_${profile}.pid"
  if [[ -f "$pid_file" ]]; then
    pid="$(tr -d '[:space:]' < "$pid_file")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      echo "G11-A1 $profile collection is still running with PID $pid" >&2
      exit 1
    fi
  fi
done

/usr/bin/python3 "$PROJECT_ROOT/scripts/audit_g11_a1_shards.py" > /dev/null

mkdir -p "$RUN_DIR/local_data/training/seed${SEED}" "$PROJECT_ROOT/logs/active/g11_a1"
log_file="$PROJECT_ROOT/logs/active/g11_a1/train_g11_a1_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

cd "$PROJECT_ROOT"
/usr/bin/time -f 'elapsed=%E maxrss_kb=%M' nice -n 15 /usr/bin/python3 \
  scripts/train_temporal_interaction_gate.py \
  --experiment-id G11-A1 \
  --train-dir "$TRAIN_DIR" \
  --validation-dir "$VALIDATION_DIR" \
  --train-manifest "$VIEW_DIR/train.json.gz" \
  --validation-manifest "$VIEW_DIR/validation.json.gz" \
  --labels any \
  --models S0 T1 \
  --threshold-policy match-s0-fpr \
  --epochs 40 \
  --seed "$SEED" \
  --device cpu \
  --output-dir "$RUN_DIR/local_data/training/seed${SEED}" \
  2>&1 | tee "$log_file"

echo "G11-A1 training log: $log_file"
