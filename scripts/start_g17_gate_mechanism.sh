#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/logs/active/g17-gate-mechanism"
PID_FILE="$ROOT/.g17_gate_mechanism.pid"
mkdir -p "$LOG_DIR"
if [[ -f "$PID_FILE" ]]; then
  pid="$(tr -d '[:space:]' <"$PID_FILE")"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    echo "G17 Gate mechanism queue already running as PID $pid"
    exit 0
  fi
  rm -f "$PID_FILE"
fi
setsid bash "$ROOT/scripts/run_g17_gate_mechanism_worker.sh" >>"$LOG_DIR/runner.log" 2>&1 < /dev/null &
echo $! >"$PID_FILE"
echo "Queued G17 Gate mechanism comparison."
echo "PID: $(cat "$PID_FILE")"
echo "Live log: $LOG_DIR/runner.log"
