#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_NAME="${DRL_MULTI_TRAIN_FILE_NAME:-current_generalist_from_e2_local_critic_s20260811}"
PID_FILE="$PROJECT_ROOT/.train_${MODEL_NAME}.pid"
if [[ ! -f "$PID_FILE" ]]; then
  echo "No current generalist from E2 local-critic PID file found: $PID_FILE"
  exit 0
fi
pid="$(tr -d '[:space:]' < "$PID_FILE")"
if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
  echo "Malformed PID file: $PID_FILE" >&2
  exit 1
fi
pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
echo "Stopping current generalist from E2 local-critic pid=$pid pgid=$pgid"
kill -TERM "$pid" 2>/dev/null || true
sleep 5
if kill -0 "$pid" 2>/dev/null; then
  kill -KILL "$pid" 2>/dev/null || true
fi
unlink "$PID_FILE" 2>/dev/null || true
echo "Stopped current generalist from E2 local-critic."
