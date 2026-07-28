#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_NAME="${DRL_MULTI_TRAIN_FILE_NAME:-independent_dense_actor_from_5a_full_v1_s20260728}"
SAFE_MODEL="${MODEL_NAME//[^A-Za-z0-9_]/_}"
PID_FILE="${DRL_MULTI_PID_FILE:-$PROJECT_ROOT/.train_${SAFE_MODEL}.pid}"

if [[ ! -f "$PID_FILE" ]]; then
  echo "Independent Dense Actor training is not running."
  exit 0
fi

pid="$(tr -d '[:space:]' < "$PID_FILE")"
if [[ ! "$pid" =~ ^[0-9]+$ ]] || ! kill -0 "$pid" 2>/dev/null; then
  unlink "$PID_FILE"
  echo "Removed stale PID file."
  exit 0
fi

kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"
echo "Stop requested for process group $pid. Latest checkpoint is preserved."
