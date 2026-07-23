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

Supported experiment IDs:
  train-strong-interaction  Current temporal strong-interaction Actor pilot
  eval-5d-standard          Historical metric-fixed 5D baseline
EOF
}

script_for() {
  local action="$1"
  local experiment_id="$2"
  case "${action}:${experiment_id}" in
    start:train-strong-interaction)
      echo "$PROJECT_ROOT/scripts/start_training_strong_interaction_expert_pilot.sh"
      ;;
    stop:train-strong-interaction)
      echo "$PROJECT_ROOT/scripts/stop_training_strong_interaction_expert_pilot.sh"
      ;;
    start:eval-5d-standard)
      echo "$PROJECT_ROOT/scripts/start_test_detached_multi_stage2_to_5d_geo_critic_from_5a_guarded_best.sh"
      ;;
    stop:eval-5d-standard)
      echo "$PROJECT_ROOT/scripts/stop_test_detached_multi_stage2_to_5d_geo_critic_from_5a_guarded_best.sh"
      ;;
    *)
      return 1
      ;;
  esac
}

show_status() {
  local found=0
  local pid_file pid
  for pid_file in \
    "$PROJECT_ROOT/.train_strong_interaction_expert_pilot.pid" \
    "$PROJECT_ROOT/.test_multi_stage2_to_5d_geo_critic_from_5a_guarded_best_detached.pid"; do
    [[ -f "$pid_file" ]] || continue
    pid="$(tr -d '[:space:]' < "$pid_file")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      printf 'running  pid=%s  pid_file=%s\n' "$pid" "${pid_file#$PROJECT_ROOT/}"
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
