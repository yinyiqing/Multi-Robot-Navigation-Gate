#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究/G11_B_student_rollout_v1"
SEED="${1:-20260804}"
LOG_NAMESPACE="${G11_B_LOG_NAMESPACE:-g11_b}"

[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "Seed must be an integer" >&2; exit 2; }
[[ "$LOG_NAMESPACE" =~ ^[a-z0-9_-]+$ ]] || {
  echo "G11_B_LOG_NAMESPACE must contain only lowercase letters, digits, _ or -" >&2
  exit 2
}
command -v flock >/dev/null 2>&1 || { echo "flock is required" >&2; exit 1; }
mkdir -p "$RUN_DIR/local_data/.locks"
exec 9>"$RUN_DIR/local_data/.locks/training_seed${SEED}.lock"
if ! flock -n 9; then
  echo "G11-B2 seed $SEED is already running" >&2
  exit 1
fi

output_dir="$RUN_DIR/local_data/training/seed${SEED}"
if [[ -f "$output_dir/summary.json" ]]; then
  echo "G11-B2 seed $SEED already has a completed summary" >&2
  exit 1
fi

/usr/bin/python3 "$PROJECT_ROOT/scripts/audit_g11_a1_shards.py" >/dev/null
/usr/bin/python3 "$PROJECT_ROOT/scripts/audit_g11_b_student_shards.py" --profile train >/dev/null

mkdir -p "$output_dir" "$PROJECT_ROOT/logs/active/$LOG_NAMESPACE"
log_file="$PROJECT_ROOT/logs/active/$LOG_NAMESPACE/train_g11_b2_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

cd "$PROJECT_ROOT"
/usr/bin/time -f 'elapsed=%E maxrss_kb=%M' nice -n 15 /usr/bin/python3 \
  scripts/train_g11_b_aggregated_gate.py \
  --epochs 40 \
  --seed "$SEED" \
  --device cpu \
  --output-dir "$output_dir" \
  2>&1 | tee "$log_file"

echo "G11-B2 training log: $log_file"
