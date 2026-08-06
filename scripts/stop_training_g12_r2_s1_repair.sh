#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$PROJECT_ROOT/.g12_r2_s1_repair.pid"
ROS_PORT=14631
GAZEBO_PORT=14731
source "$PROJECT_ROOT/scripts/lib_g12_r2_s1_runtime.sh"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No managed G12-R2-S1 repair pilot is running."
  exit 0
fi

pid="$(tr -d '[:space:]' < "$PID_FILE")"
if [[ "$pid" =~ ^[0-9]+$ ]]; then
  g12_r2_s1_stop_stage "$pid" "$ROS_PORT" "$GAZEBO_PORT" || true
fi
unlink "$PID_FILE" 2>/dev/null || true

if ! g12_r2_s1_ports_are_free "$ROS_PORT" "$GAZEBO_PORT"; then
  echo "S1 repair ROS/Gazebo ports are still occupied" >&2
  exit 1
fi
echo "Stopped G12-R2-S1 repair pilot."
