#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$PROJECT_ROOT/.train_fixed_v1_edge1_residual_pilot.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No edge-1 residual pilot PID file found."
  exit 0
fi

pid="$(cat "$PID_FILE")"
if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
  kill -TERM -- "-$pid"
  echo "Stopped edge-1 residual pilot process group $pid."
else
  echo "Edge-1 residual pilot PID $pid is not running."
fi
rm -f "$PID_FILE"
