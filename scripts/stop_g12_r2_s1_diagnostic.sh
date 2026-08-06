#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$PROJECT_ROOT/.g12_r2_s1_diagnostic.pid"
ROS_PORT=14621
GAZEBO_PORT=14721
source "$PROJECT_ROOT/scripts/lib_g12_r2_s1_runtime.sh"

if [[ -f "$PID_FILE" ]]; then
  pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 45); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
    echo "Stopped G12-R2-S1 diagnostic worker $pid."
  else
    echo "Managed G12-R2-S1 diagnostic is not active."
  fi
  unlink "$PID_FILE"
else
  echo "No managed G12-R2-S1 diagnostic is running."
fi

for stage in stage1_single stage1e_single_rescue stage1f_wall_parallel_rescue stage1g_collision_guard; do
  stage_pid_file="$PROJECT_ROOT/.test_multi_curriculum_${stage}_detached.pid"
  [[ -f "$stage_pid_file" ]] || continue
  stage_pid="$(tr -d '[:space:]' < "$stage_pid_file")"
  g12_r2_s1_stop_stage "$stage_pid" "$ROS_PORT" "$GAZEBO_PORT" || true
  unlink "$stage_pid_file" 2>/dev/null || true
done

if ! g12_r2_s1_ports_are_free "$ROS_PORT" "$GAZEBO_PORT"; then
  echo "Warning: S1 ROS/Gazebo ports are still occupied" >&2
  exit 1
fi
