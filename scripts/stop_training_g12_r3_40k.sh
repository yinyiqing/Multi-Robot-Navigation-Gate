#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$PROJECT_ROOT/.g12_r3_40k.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "G12-R3 is not running."
  exit 0
fi
pid="$(tr -d '[:space:]' < "$PID_FILE")"
if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  echo "Stopped G12-R3 process group $pid."
else
  echo "Removed stale G12-R3 PID file."
fi
unlink "$PID_FILE" 2>/dev/null || true
