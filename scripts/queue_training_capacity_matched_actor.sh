#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUEUE_PID_FILE="$PROJECT_ROOT/.g12_capacity_queue.pid"
D2_PID_FILE="$PROJECT_ROOT/.g11_d2_admission.pid"
D2_SUMMARY="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究/G11_D_Gate复核与独立准入/local_data/d2_summary.json"
LOG_DIR="$PROJECT_ROOT/logs/active/capacity-matched-actor-g12-p1"
QUEUE_LOG="$LOG_DIR/queue.log"
MIN_FREE_GPU_MIB="${G12_MIN_FREE_GPU_MIB:-8192}"
MAX_GPU_UTILIZATION="${G12_MAX_GPU_UTILIZATION:-20}"

run_worker() {
  cleanup() {
    unlink "$QUEUE_PID_FILE" 2>/dev/null || true
  }
  trap cleanup EXIT

  echo "[$(date --iso-8601=seconds)] G12 queue is waiting for the G11-D2 archive."
  while [[ -f "$D2_PID_FILE" ]]; do
    d2_pid="$(tr -d '[:space:]' < "$D2_PID_FILE")"
    if [[ ! "$d2_pid" =~ ^[0-9]+$ ]] || ! kill -0 "$d2_pid" 2>/dev/null; then
      echo "[$(date --iso-8601=seconds)] G11-D2 left a stale PID file; refusing automatic launch."
      exit 1
    fi
    sleep 60
  done

  if [[ ! -f "$D2_SUMMARY" ]]; then
    echo "[$(date --iso-8601=seconds)] G11-D2 PID ended without d2_summary.json; refusing automatic launch."
    exit 1
  fi

  command -v nvidia-smi >/dev/null 2>&1 || {
    echo "[$(date --iso-8601=seconds)] nvidia-smi is unavailable; refusing to start GPU-required G12."
    exit 1
  }

  while true; do
    gpu_state="$(nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)"
    free_mib="$(printf '%s' "$gpu_state" | cut -d, -f1 | tr -d '[:space:]')"
    utilization="$(printf '%s' "$gpu_state" | cut -d, -f2 | tr -d '[:space:]')"
    if [[ "$free_mib" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]] \
      && (( free_mib >= MIN_FREE_GPU_MIB && utilization <= MAX_GPU_UTILIZATION )); then
      break
    fi
    echo "[$(date --iso-8601=seconds)] GPU 0 not ready (free=${free_mib:-unknown} MiB, utilization=${utilization:-unknown}%); waiting."
    sleep 60
  done

  export CUDA_VISIBLE_DEVICES=0
  echo "[$(date --iso-8601=seconds)] GPU 0 ready (free=${free_mib} MiB, utilization=${utilization}%); G12 will use CUDA."

  echo "[$(date --iso-8601=seconds)] G11-D2 archive verified; starting G12-P1."
  bash "$PROJECT_ROOT/scripts/start_training_capacity_matched_actor.sh"
}

if [[ "${1:-}" == "worker" ]]; then
  run_worker
  exit 0
fi

if [[ -f "$QUEUE_PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$QUEUE_PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "G12 capacity queue is already running with PID $old_pid"
    exit 1
  fi
  unlink "$QUEUE_PID_FILE"
fi
if [[ -f "$PROJECT_ROOT/.g12_capacity_actor.pid" ]]; then
  echo "G12 capacity training already owns its PID file."
  exit 1
fi

mkdir -p "$LOG_DIR"
setsid bash "$0" worker >>"$QUEUE_LOG" 2>&1 < /dev/null &
echo $! > "$QUEUE_PID_FILE"
echo "Queued G12-P1 after G11-D2 archival."
echo "Queue PID: $(cat "$QUEUE_PID_FILE")"
echo "Queue log: $QUEUE_LOG"
