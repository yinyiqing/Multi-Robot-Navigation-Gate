#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$PROJECT_ROOT/.test_multi_stage2_to_5d_geo_critic_from_5a_guarded_best_detached.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No detached stage2-to-5D geometry-critic best test pid file found."
  exit 0
fi

pid="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ "$pid" =~ ^[0-9]+$ ]]; then
  # The process-group leader may exit before roslaunch/Gazebo. The PGID remains
  # addressable, so always try the group before treating the PID as stale.
  if kill -- "-$pid" 2>/dev/null; then
    sleep 2
    kill -9 -- "-$pid" 2>/dev/null || true
  elif kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
  fi
  echo "Stopped detached stage2-to-5D geometry-critic best test process group led by PID $pid"
else
  echo "Invalid or empty PID file: $PID_FILE"
fi

rm -f "$PID_FILE"
