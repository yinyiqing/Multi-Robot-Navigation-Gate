#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究/G11_D_Gate复核与独立准入"
ACTIVE_LOG_DIR="$PROJECT_ROOT/logs/active/g11_d"
ARCHIVE_LOG_DIR="$PROJECT_ROOT/logs/archive/diagnostic/g11_d"
SEEDS=(20260805 20260806 20260807 20260808)

command -v flock >/dev/null 2>&1 || { echo "flock is required" >&2; exit 1; }
mkdir -p "$RUN_DIR/local_data/.locks" "$ACTIVE_LOG_DIR"
exec 9>"$RUN_DIR/local_data/.locks/seed_replication.lock"
if ! flock -n 9; then
  echo "G11-D1 seed replication is already running" >&2
  exit 1
fi

driver_log="$ACTIVE_LOG_DIR/seed_replication_driver_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$driver_log") 2>&1

echo "G11-D1 started at $(date --iso-8601=seconds)"
echo "CPU-only sequential seeds: ${SEEDS[*]}"
for seed in "${SEEDS[@]}"; do
  summary="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究/G11_B_student_rollout_v1/local_data/training/seed${seed}/summary.json"
  if [[ -f "$summary" ]]; then
    echo "seed $seed already complete; preserving existing output"
    continue
  fi
  G11_B_LOG_NAMESPACE=g11_d \
    bash "$PROJECT_ROOT/scripts/run_g11_b_aggregated_training.sh" "$seed"
done

/usr/bin/python3 "$PROJECT_ROOT/scripts/analyze_g11_d_seed_replication.py"
echo "G11-D1 completed at $(date --iso-8601=seconds)"

mkdir -p "$ARCHIVE_LOG_DIR"
while IFS= read -r -d '' log_file; do
  mv -n "$log_file" "$ARCHIVE_LOG_DIR/"
done < <(find "$ACTIVE_LOG_DIR" -maxdepth 1 -type f -name '*.log' -print0)
rmdir "$ACTIVE_LOG_DIR" 2>/dev/null || true
echo "G11-D1 logs archived to $ARCHIVE_LOG_DIR"
