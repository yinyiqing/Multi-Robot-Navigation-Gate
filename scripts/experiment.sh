#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/experiment.sh list
  bash scripts/experiment.sh status
  bash scripts/experiment.sh start actor-g12-r2-s1-repair
  bash scripts/experiment.sh stop actor-g12-r2-s1-repair

Current method:
  Actor N  generalist-5a         frozen
  Actor I  interaction-epoch16   frozen
  Gate     G11-B2                 D2 navigation passed, efficiency failed

Current command:
  actor-g12-r2-s1-repair  S0 full warm start, 8 repair cases, first 20k pilot

G11-C, G11-D2, G12-P1 and G12-R1 are complete. Their start/stop entrypoints are
retained only for controlled reproduction.

G12-R2-S0 and the S1 diagnostic are complete. The repair pilot is the only current
Actor-training command. Historical scripts are not current execution instructions.
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
      elif [[ "$(basename "$pid_file")" == ".g11_d2_admission.pid" ]]; then
        printf '  logs=logs/active/gate-g11-d2-admission/'
      elif [[ "$(basename "$pid_file")" == ".g12_capacity_actor.pid" ]]; then
        printf '  logs=logs/active/capacity-matched-actor-g12-p1/'
      elif [[ "$(basename "$pid_file")" == ".g12_capacity_queue.pid" ]]; then
        printf '  state=waiting-for-d2  logs=logs/active/capacity-matched-actor-g12-p1/queue.log'
      elif [[ "$(basename "$pid_file")" == ".g12_capacity_r1.pid" ]]; then
        printf '  logs=logs/active/capacity-original-width-r1/'
      elif [[ "$(basename "$pid_file")" == ".g12_r2_s0.pid" ]]; then
        printf '  logs=logs/active/capacity-wide-g12-r2/'
      elif [[ "$(basename "$pid_file")" == ".g12_r2_s1_diagnostic.pid" ]]; then
        printf '  logs=logs/active/capacity-wide-g12-r2/s1-diagnostic/'
      elif [[ "$(basename "$pid_file")" == ".g12_r2_s1_repair.pid" ]]; then
        printf '  logs=logs/active/capacity-wide-g12-r2/s1-repair/'
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
  queue)
    case "${2:-}" in
      actor-g12-capacity-pilot) exec bash "$PROJECT_ROOT/scripts/queue_training_capacity_matched_actor.sh" ;;
      *) usage >&2; exit 2 ;;
    esac
    ;;
  start)
    case "${2:-}" in
      gate-g11-b-smoke) exec bash "$PROJECT_ROOT/scripts/start_g11_b_student_collection.sh" smoke ;;
      gate-g11-b-train) exec bash "$PROJECT_ROOT/scripts/start_g11_b_student_collection.sh" train ;;
      gate-g11-c-pilot) exec bash "$PROJECT_ROOT/scripts/start_g11_c_pilot.sh" ;;
      gate-g11-d2-admission) exec bash "$PROJECT_ROOT/scripts/start_g11_d2_admission.sh" ;;
      actor-g12-capacity-pilot) exec bash "$PROJECT_ROOT/scripts/start_training_capacity_matched_actor.sh" ;;
      actor-g12-r1-original-width) exec bash "$PROJECT_ROOT/scripts/start_training_capacity_original_width_control.sh" ;;
      actor-g12-r2-s0) exec bash "$PROJECT_ROOT/scripts/start_training_g12_r2_s0.sh" ;;
      actor-g12-r2-s1-diagnostic) exec bash "$PROJECT_ROOT/scripts/start_g12_r2_s1_diagnostic.sh" ;;
      actor-g12-r2-s1-repair) exec bash "$PROJECT_ROOT/scripts/start_training_g12_r2_s1_repair.sh" ;;
      *) usage >&2; exit 2 ;;
    esac
    ;;
  stop)
    case "${2:-}" in
      gate-g11-b-smoke) exec bash "$PROJECT_ROOT/scripts/stop_g11_b_student_collection.sh" smoke ;;
      gate-g11-b-train) exec bash "$PROJECT_ROOT/scripts/stop_g11_b_student_collection.sh" train ;;
      gate-g11-c-pilot) exec bash "$PROJECT_ROOT/scripts/stop_g11_c_pilot.sh" ;;
      gate-g11-d2-admission) exec bash "$PROJECT_ROOT/scripts/stop_g11_d2_admission.sh" ;;
      actor-g12-capacity-pilot) exec bash "$PROJECT_ROOT/scripts/stop_training_capacity_matched_actor.sh" ;;
      actor-g12-r1-original-width) exec bash "$PROJECT_ROOT/scripts/stop_training_capacity_original_width_control.sh" ;;
      actor-g12-r2-s0) exec bash "$PROJECT_ROOT/scripts/stop_training_g12_r2_s0.sh" ;;
      actor-g12-r2-s1-diagnostic) exec bash "$PROJECT_ROOT/scripts/stop_g12_r2_s1_diagnostic.sh" ;;
      actor-g12-r2-s1-repair) exec bash "$PROJECT_ROOT/scripts/stop_training_g12_r2_s1_repair.sh" ;;
      *) usage >&2; exit 2 ;;
    esac
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
