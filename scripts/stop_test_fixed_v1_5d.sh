#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POOL="${1:-}"
if [[ "$POOL" != "standard" && "$POOL" != "dense" ]]; then
  echo "Usage: $0 standard|dense"
  exit 2
fi

PID_FILE="$PROJECT_ROOT/.test_fixed_v1_${POOL}_5d.pid"
if [[ ! -f "$PID_FILE" ]]; then
  echo "No fixed-v1 $POOL test PID file found."
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
  echo "Stopped fixed-v1 $POOL test process group led by PID $pid"
else
  echo "Invalid PID file: $PID_FILE"
fi
rm -f "$PID_FILE"
