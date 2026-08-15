#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$PROJECT_ROOT/.g12_r2c_corrected.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "R2C corrected continuation is not running."
  exit 0
fi
pid="$(tr -d '[:space:]' < "$PID_FILE")"
if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
  pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
  kill -TERM -- "-$pgid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  echo "Sent TERM to R2C corrected continuation process group $pgid."
else
  unlink "$PID_FILE"
  echo "Removed stale R2C corrected continuation PID file."
fi
