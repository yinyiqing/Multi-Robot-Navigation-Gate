#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
G27="$BASE/27_反事实Reward增强Gate监督"
PID_FILE="$ROOT/.g27_p1_collection.pid"
LOG_DIR="$G27/logs"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "G27 P1 collection is already running with PID $old_pid" >&2
    exit 1
  fi
  unlink "$PID_FILE"
fi

mkdir -p "$LOG_DIR"
timestamp="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/p1_collection_${timestamp}.log"

setsid bash -lc "
  set -eo pipefail
  cleanup() { unlink '$PID_FILE' 2>/dev/null || true; }
  trap cleanup EXIT
  bash '$ROOT/scripts/run_g27_p1_collection.sh' train
  bash '$ROOT/scripts/run_g27_p1_collection.sh' validation
" >"$LOG_FILE" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "Started G27 P1 train -> validation collection queue."
echo "PID: $(cat "$PID_FILE")"
echo "Log: $LOG_FILE"
