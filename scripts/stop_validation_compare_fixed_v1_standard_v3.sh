#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$PROJECT_ROOT/.validation_fixed_v1_standard_v3.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No validation comparison PID file found."
  exit 0
fi

pid="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ "$pid" =~ ^[0-9]+$ ]]; then
  kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
  sleep 2
  kill -9 -- "-$pid" 2>/dev/null || true
  echo "Stopped validation comparison process group led by PID $pid"
fi
rm -f "$PID_FILE"
