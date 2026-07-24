#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$PROJECT_ROOT/.test_lidar_cluster_sensor_probe_5d.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No managed lidar cluster sensor probe is running."
  exit 0
fi

pid="$(tr -d '[:space:]' < "$PID_FILE")"
if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
  pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
  kill -- "-$pgid" 2>/dev/null || kill "$pid" 2>/dev/null || true
  echo "Stopped lidar cluster sensor probe process group $pgid."
else
  echo "Managed lidar cluster sensor probe is not active."
fi
unlink "$PID_FILE"
