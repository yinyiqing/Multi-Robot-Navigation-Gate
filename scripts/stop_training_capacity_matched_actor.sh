#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUEUE_PID_FILE="$PROJECT_ROOT/.g12_capacity_queue.pid"
if [[ -f "$QUEUE_PID_FILE" ]]; then
  queue_pid="$(tr -d '[:space:]' < "$QUEUE_PID_FILE")"
  if [[ "$queue_pid" =~ ^[0-9]+$ ]] && kill -0 "$queue_pid" 2>/dev/null; then
    kill -TERM -- "-$queue_pid" 2>/dev/null || kill -TERM "$queue_pid"
    echo "Cancelled G12 capacity queue $queue_pid."
  else
    unlink "$QUEUE_PID_FILE"
  fi
fi
export DRL_MULTI_TRAIN_FILE_NAME=capacity_matched_actor_wide_n5_seed20260810_pilot
export DRL_MULTI_EXPERIMENT_LABEL=G12-P1-capacity-matched-wide-actor
export DRL_MULTI_PID_FILE="$PROJECT_ROOT/.g12_capacity_actor.pid"
exec "$PROJECT_ROOT/scripts/stop_training_dense_simple_td3_hparam_a.sh"
