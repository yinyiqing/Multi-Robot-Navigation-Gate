#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:-stage1_single}"
PID_FILE="$PROJECT_ROOT/.test_multi_curriculum_${STAGE}_detached.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No detached curriculum test PID file found for stage: $STAGE"
  exit 0
fi

pid="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
  echo "Invalid or empty PID file for stage: $STAGE"
  rm -f "$PID_FILE"
  exit 0
fi

if kill -- "-$pid" 2>/dev/null; then
  sleep 2
  kill -9 -- "-$pid" 2>/dev/null || true
elif kill -0 "$pid" 2>/dev/null; then
  kill "$pid" 2>/dev/null || true
fi
rm -f "$PID_FILE"
echo "Stopped detached curriculum test process group led by PID $pid"
