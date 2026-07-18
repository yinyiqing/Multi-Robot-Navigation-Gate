#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$PROJECT_ROOT/.train_fixed_v1_standard_expert.pid"
if [[ ! -f "$PID_FILE" ]]; then
  echo "No standard expert training PID file found."
  exit 0
fi
pid="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ "$pid" =~ ^[0-9]+$ ]]; then
  if kill -- "-$pid" 2>/dev/null; then
    sleep 2
    kill -9 -- "-$pid" 2>/dev/null || true
  elif kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
  fi
  echo "Stopped standard expert training process group led by PID $pid"
fi
rm -f "$PID_FILE"
