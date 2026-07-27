#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/experiment.sh list
  bash scripts/experiment.sh status
  bash scripts/experiment.sh start <experiment-id>
  bash scripts/experiment.sh stop <experiment-id>

Supported current experiment IDs:
  (none; gate-robot-perception is pending its G0 data protocol)
EOF
}

script_for() {
  return 1
}

show_status() {
  local found=0
  local pid_file pid
  for pid_file in \
    "$PROJECT_ROOT/.test_lidar_cluster_sensor_probe_5d.pid"; do
    [[ -f "$pid_file" ]] || continue
    pid="$(tr -d '[:space:]' < "$pid_file")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      printf 'legacy-running  pid=%s  pid_file=%s\n' "$pid" "${pid_file#$PROJECT_ROOT/}"
      found=1
    else
      printf 'stale    pid=%s  pid_file=%s\n' "${pid:-invalid}" "${pid_file#$PROJECT_ROOT/}"
      found=1
    fi
  done
  if [[ "$found" -eq 0 ]]; then
    echo "No managed experiment is running."
  fi
}

command="${1:-}"
case "$command" in
  list)
    usage
    ;;
  status)
    show_status
    ;;
  start|stop)
    experiment_id="${2:-}"
    if [[ -z "$experiment_id" ]]; then
      usage >&2
      exit 2
    fi
    if ! script="$(script_for "$command" "$experiment_id")"; then
      echo "Unknown or unsupported experiment ID: $experiment_id" >&2
      usage >&2
      exit 2
    fi
    exec "$script"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
