#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$PROJECT_ROOT/.current_generalist_n5_admission.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No current-generalist N5 admission PID file found: $PID_FILE"
  exit 0
fi

pid="$(tr -d '[:space:]' < "$PID_FILE")"
if [[ ! "$pid" =~ ^[0-9]+$ ]] || ! kill -0 "$pid" 2>/dev/null; then
  echo "PID file is stale: $PID_FILE"
  unlink "$PID_FILE"
  exit 0
fi

pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
echo "Stopping current-generalist N5 admission pid=$pid pgid=$pgid"
kill -TERM "-$pgid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
sleep 5
if kill -0 "$pid" 2>/dev/null; then
  echo "Process still alive; sending KILL to process group"
  kill -KILL "-$pgid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
fi
unlink "$PID_FILE" 2>/dev/null || true
