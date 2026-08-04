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
  Gate     G11-A1 temporal Gate  data collection ready

Current commands:
  bash scripts/experiment.sh start gate-g11-a1-train
  bash scripts/experiment.sh start gate-g11-a1-validation
  bash scripts/experiment.sh stop  gate-g11-a1-train
  bash scripts/experiment.sh stop  gate-g11-a1-validation

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
      printf 'legacy-running  pid=%s  pid_file=%s\n' \
        "$pid" "${pid_file#$PROJECT_ROOT/}"
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
      gate-g11-a1-train) exec bash "$PROJECT_ROOT/scripts/start_g11_a1_collection.sh" train ;;
      gate-g11-a1-validation) exec bash "$PROJECT_ROOT/scripts/start_g11_a1_collection.sh" validation ;;
      *) usage >&2; exit 2 ;;
    esac
    ;;
  stop)
    case "${2:-}" in
      gate-g11-a1-train) exec bash "$PROJECT_ROOT/scripts/stop_g11_a1_collection.sh" train ;;
      gate-g11-a1-validation) exec bash "$PROJECT_ROOT/scripts/stop_g11_a1_collection.sh" validation ;;
      *) usage >&2; exit 2 ;;
    esac
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
