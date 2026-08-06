#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$PROJECT_ROOT/.g12_r2_s0.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No G12-R2-S0 PID file found."
  exit 0
fi
pid="$(tr -d '[:space:]' < "$PID_FILE")"
if [[ ! "$pid" =~ ^[0-9]+$ ]] || ! kill -0 "$pid" 2>/dev/null; then
  echo "G12-R2-S0 is not running; removing stale PID file."
  unlink "$PID_FILE"
  exit 0
fi

pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
kill -TERM -- "-$pgid" 2>/dev/null || true
for _ in $(seq 1 20); do
  kill -0 "$pid" 2>/dev/null || break
  sleep 0.5
done
if kill -0 "$pid" 2>/dev/null; then
  kill -KILL -- "-$pgid" 2>/dev/null || true
fi
unlink "$PID_FILE" 2>/dev/null || true
echo "Stopped G12-R2-S0 process group $pgid."
