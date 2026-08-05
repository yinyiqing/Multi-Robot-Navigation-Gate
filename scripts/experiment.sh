#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/experiment.sh list
  bash scripts/experiment.sh status

Current method:
  Actor N  generalist-5a         frozen
  Actor I  interaction-epoch16   frozen
  Gate     G11-B2                 trained, closed-loop pilot running

Current commands:
  bash scripts/experiment.sh start gate-g11-c-pilot
  bash scripts/experiment.sh stop  gate-g11-c-pilot

Actor training remains closed. Historical scripts are not current entrypoints.
EOF
}

show_status() {
  local found=0
  local pid_file pid
  while IFS= read -r pid_file; do
    [[ -f "$pid_file" ]] || continue
    pid="$(tr -d '[:space:]' < "$pid_file")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      printf 'running         pid=%s  pid_file=%s' \
        "$pid" "${pid_file#$PROJECT_ROOT/}"
      if [[ "$(basename "$pid_file")" == ".g11_c_pilot.pid" ]]; then
        printf '  logs=logs/active/gate-g11-c-pilot/'
      fi
      printf '\n'
    else
      printf 'stale           pid=%s  pid_file=%s\n' \
        "${pid:-invalid}" "${pid_file#$PROJECT_ROOT/}"
    fi
    found=1
  done < <(find "$PROJECT_ROOT" -maxdepth 1 -type f -name '.*.pid' -print | sort)
  if [[ "$found" -eq 0 ]]; then
    echo "No managed experiment is running."
  fi
}

case "${1:-}" in
  list)
    usage
    ;;
  status)
    show_status
    ;;
  start)
    case "${2:-}" in
      gate-g11-b-smoke) exec bash "$PROJECT_ROOT/scripts/start_g11_b_student_collection.sh" smoke ;;
      gate-g11-b-train) exec bash "$PROJECT_ROOT/scripts/start_g11_b_student_collection.sh" train ;;
      gate-g11-c-pilot) exec bash "$PROJECT_ROOT/scripts/start_g11_c_pilot.sh" ;;
      *) usage >&2; exit 2 ;;
    esac
    ;;
  stop)
    case "${2:-}" in
      gate-g11-b-smoke) exec bash "$PROJECT_ROOT/scripts/stop_g11_b_student_collection.sh" smoke ;;
      gate-g11-b-train) exec bash "$PROJECT_ROOT/scripts/stop_g11_b_student_collection.sh" train ;;
      gate-g11-c-pilot) exec bash "$PROJECT_ROOT/scripts/stop_g11_c_pilot.sh" ;;
      *) usage >&2; exit 2 ;;
    esac
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
