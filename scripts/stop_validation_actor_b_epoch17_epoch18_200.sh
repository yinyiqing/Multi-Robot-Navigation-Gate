#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$PROJECT_ROOT/.validation_actor_b_epoch17_epoch18_200.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No Actor B paired validation PID file found."
  exit 0
fi

pid="$(tr -d '[:space:]' < "$PID_FILE")"
if [[ "$pid" =~ ^[0-9]+$ ]]; then
  if kill -- "-$pid" 2>/dev/null; then
    sleep 2
    kill -9 -- "-$pid" 2>/dev/null || true
  elif kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
  fi
  echo "Stopped Actor B paired validation process group led by PID $pid."
else
  echo "Invalid PID file: $PID_FILE"
fi
rm -f "$PID_FILE"
