#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/experiment.sh list
  bash scripts/experiment.sh status

Current experiment:
  actor-b-single-edge-residual  approved, implementation not yet admitted

No start/stop command is exposed until the R0/R1 short-pilot protocol and
implementation checks are complete. Historical scripts are not current entrypoints.
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
  *)
    usage >&2
    exit 2
    ;;
esac
