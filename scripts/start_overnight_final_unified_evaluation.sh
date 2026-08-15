#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/logs/active/g20-overnight-final-unified"
PID_FILE="$ROOT/.g20_overnight_final_unified.pid"

if [[ -f "$PID_FILE" ]]; then
  pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    echo "G20 is already queued or running as PID $pid" >&2
    exit 1
  fi
  unlink "$PID_FILE"
fi

mkdir -p "$LOG_DIR"
setsid bash "$ROOT/scripts/run_overnight_final_unified_evaluation_worker.sh" \
  >>"$LOG_DIR/runner.log" 2>&1 < /dev/null &
echo $! > "$PID_FILE"
echo "Queued G20 overnight final unified evaluation"
echo "PID: $(cat "$PID_FILE")"
echo "Live log: $LOG_DIR/runner.log"
