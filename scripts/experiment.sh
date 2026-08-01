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
  gate-robot-perception-train
  gate-robot-perception-validation
  gate-robot-perception-pilot-train
  gate-robot-perception-pilot-validation
  gate-robot-tracking-pilot-train
  gate-robot-tracking-pilot-validation
  gate-interaction-pilot-train
  gate-interaction-pilot-validation
  gate-counterfactual-pilot-train
  gate-counterfactual-pilot-validation
  learned-gate-validation
  g4-counterfactual-smoke
  g4-counterfactual-pilot
  edge2-confirmation-5a
  edge2-confirmation-learned-gate
EOF
}

script_for() {
  local command="$1"
  local experiment_id="$2"
  case "$command:$experiment_id" in
    start:gate-robot-perception-train)
      echo "$PROJECT_ROOT/scripts/start_robot_perception_collection.sh train"
      ;;
    stop:gate-robot-perception-train)
      echo "$PROJECT_ROOT/scripts/stop_robot_perception_collection.sh train"
      ;;
    start:gate-robot-perception-validation)
      echo "$PROJECT_ROOT/scripts/start_robot_perception_collection.sh validation"
      ;;
    stop:gate-robot-perception-validation)
      echo "$PROJECT_ROOT/scripts/stop_robot_perception_collection.sh validation"
      ;;
    start:gate-robot-perception-pilot-train)
      echo "$PROJECT_ROOT/scripts/start_robot_perception_collection.sh pilot-train"
      ;;
    stop:gate-robot-perception-pilot-train)
      echo "$PROJECT_ROOT/scripts/stop_robot_perception_collection.sh pilot-train"
      ;;
    start:gate-robot-perception-pilot-validation)
      echo "$PROJECT_ROOT/scripts/start_robot_perception_collection.sh pilot-validation"
      ;;
    stop:gate-robot-perception-pilot-validation)
      echo "$PROJECT_ROOT/scripts/stop_robot_perception_collection.sh pilot-validation"
      ;;
    start:gate-robot-tracking-pilot-train)
      echo "$PROJECT_ROOT/scripts/start_robot_perception_collection.sh tracking-pilot-train"
      ;;
    stop:gate-robot-tracking-pilot-train)
      echo "$PROJECT_ROOT/scripts/stop_robot_perception_collection.sh tracking-pilot-train"
      ;;
    start:gate-robot-tracking-pilot-validation)
      echo "$PROJECT_ROOT/scripts/start_robot_perception_collection.sh tracking-pilot-validation"
      ;;
    stop:gate-robot-tracking-pilot-validation)
      echo "$PROJECT_ROOT/scripts/stop_robot_perception_collection.sh tracking-pilot-validation"
      ;;
    start:gate-interaction-pilot-train)
      echo "$PROJECT_ROOT/scripts/start_robot_perception_collection.sh gate-pilot-train"
      ;;
    stop:gate-interaction-pilot-train)
      echo "$PROJECT_ROOT/scripts/stop_robot_perception_collection.sh gate-pilot-train"
      ;;
    start:gate-interaction-pilot-validation)
      echo "$PROJECT_ROOT/scripts/start_robot_perception_collection.sh gate-pilot-validation"
      ;;
    stop:gate-interaction-pilot-validation)
      echo "$PROJECT_ROOT/scripts/stop_robot_perception_collection.sh gate-pilot-validation"
      ;;
    start:gate-counterfactual-pilot-train)
      echo "$PROJECT_ROOT/scripts/start_actor_counterfactual_collection.sh pilot-train"
      ;;
    stop:gate-counterfactual-pilot-train)
      echo "$PROJECT_ROOT/scripts/stop_actor_counterfactual_collection.sh pilot-train"
      ;;
    start:gate-counterfactual-pilot-validation)
      echo "$PROJECT_ROOT/scripts/start_actor_counterfactual_collection.sh pilot-validation"
      ;;
    stop:gate-counterfactual-pilot-validation)
      echo "$PROJECT_ROOT/scripts/stop_actor_counterfactual_collection.sh pilot-validation"
      ;;
    start:learned-gate-validation)
      echo "$PROJECT_ROOT/scripts/start_validation_learned_gate.sh"
      ;;
    stop:learned-gate-validation)
      echo "$PROJECT_ROOT/scripts/stop_validation_learned_gate.sh"
      ;;
    start:g4-counterfactual-smoke)
      echo "$PROJECT_ROOT/scripts/start_actor_counterfactual_distribution_pilot.sh smoke"
      ;;
    stop:g4-counterfactual-smoke)
      echo "$PROJECT_ROOT/scripts/stop_actor_counterfactual_distribution_pilot.sh smoke"
      ;;
    start:g4-counterfactual-pilot)
      echo "$PROJECT_ROOT/scripts/start_actor_counterfactual_distribution_pilot.sh pilot"
      ;;
    stop:g4-counterfactual-pilot)
      echo "$PROJECT_ROOT/scripts/stop_actor_counterfactual_distribution_pilot.sh pilot"
      ;;
    start:edge2-confirmation-5a)
      echo "$PROJECT_ROOT/scripts/start_exact_edge2_confirmation.sh 5a"
      ;;
    stop:edge2-confirmation-5a)
      echo "$PROJECT_ROOT/scripts/stop_exact_edge2_confirmation.sh 5a"
      ;;
    start:edge2-confirmation-learned-gate)
      echo "$PROJECT_ROOT/scripts/start_exact_edge2_confirmation.sh learned-gate"
      ;;
    stop:edge2-confirmation-learned-gate)
      echo "$PROJECT_ROOT/scripts/stop_exact_edge2_confirmation.sh learned-gate"
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
    "$PROJECT_ROOT/.test_lidar_cluster_sensor_probe_5d.pid" \
    "$PROJECT_ROOT/.robot_perception_collection_train.pid" \
    "$PROJECT_ROOT/.robot_perception_collection_validation.pid" \
    "$PROJECT_ROOT/.robot_perception_collection_pilot_train.pid" \
    "$PROJECT_ROOT/.robot_perception_collection_pilot_validation.pid" \
    "$PROJECT_ROOT/.robot_perception_collection_tracking_pilot_train.pid" \
    "$PROJECT_ROOT/.robot_perception_collection_tracking_pilot_validation.pid" \
    "$PROJECT_ROOT/.robot_perception_collection_gate_pilot_train.pid" \
    "$PROJECT_ROOT/.robot_perception_collection_gate_pilot_validation.pid" \
    "$PROJECT_ROOT/.actor_counterfactual_pilot_train.pid" \
    "$PROJECT_ROOT/.actor_counterfactual_pilot_validation.pid" \
    "$PROJECT_ROOT/.validation_learned_gate.pid" \
    "$PROJECT_ROOT/.g4_counterfactual_distribution_smoke.pid" \
    "$PROJECT_ROOT/.g4_counterfactual_distribution_pilot.pid" \
    "$PROJECT_ROOT/.edge2_confirmation_5a.pid" \
    "$PROJECT_ROOT/.edge2_confirmation_learned_gate.pid"; do
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
    read -r -a script_parts <<< "$script"
    exec "${script_parts[@]}"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
