#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_NAME="${DRL_MULTI_TRAIN_FILE_NAME:-actor_b_from_epoch16_full_pilot_v1_s20260802}"
SAFE_MODEL="${MODEL_NAME//[^A-Za-z0-9_]/_}"
PID_FILE="${DRL_MULTI_PID_FILE:-$PROJECT_ROOT/.train_${SAFE_MODEL}.pid}"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No Actor B pilot PID file found."
  exit 0
fi
pid="$(tr -d '[:space:]' < "$PID_FILE")"
if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
  kill -- "-$pid" 2>/dev/null || kill "$pid"
  echo "Stopped Actor B pilot process group $pid."
else
  echo "Actor B pilot PID ${pid:-invalid} is not running."
fi
unlink "$PID_FILE" 2>/dev/null || true
