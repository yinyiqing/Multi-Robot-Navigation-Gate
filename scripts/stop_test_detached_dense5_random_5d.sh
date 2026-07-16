#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$PROJECT_ROOT/.test_dense5_random_5d_detached.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No detached random-dense 5D test PID file found."
  exit 0
fi

pid="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
  echo "No running random-dense 5D test process found for PID $pid"
  rm -f "$PID_FILE"
  exit 0
fi

pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
if [[ -n "$pgid" ]]; then
  kill -- "-$pgid" 2>/dev/null || true
  sleep 2
  kill -9 -- "-$pgid" 2>/dev/null || true
else
  kill "$pid" 2>/dev/null || true
fi

rm -f "$PID_FILE"
echo "Stopped detached random-dense 5D test process group led by PID $pid"
