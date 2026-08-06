#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$PROJECT_ROOT/.g12_r2_s1_diagnostic.pid"

for stage in stage1_single stage1e_single_rescue stage1f_wall_parallel_rescue stage1g_collision_guard; do
  stage_pid_file="$PROJECT_ROOT/.test_multi_curriculum_${stage}_detached.pid"
  [[ -f "$stage_pid_file" ]] || continue
  stage_pid="$(tr -d '[:space:]' < "$stage_pid_file")"
  if [[ "$stage_pid" =~ ^[0-9]+$ ]] && kill -0 "$stage_pid" 2>/dev/null; then
    kill -- "-$stage_pid" 2>/dev/null || kill "$stage_pid" 2>/dev/null || true
  fi
  unlink "$stage_pid_file" 2>/dev/null || true
done

if [[ -f "$PID_FILE" ]]; then
  pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
    kill -- "-$pgid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    echo "Stopped G12-R2-S1 diagnostic process group $pgid."
  else
    echo "Managed G12-R2-S1 diagnostic is not active."
  fi
  unlink "$PID_FILE"
else
  echo "No managed G12-R2-S1 diagnostic is running."
fi
