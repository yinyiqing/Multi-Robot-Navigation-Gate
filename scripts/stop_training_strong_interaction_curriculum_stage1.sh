#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$PROJECT_ROOT/.train_strong_interaction_curriculum_stage1.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No strong-interaction curriculum Stage 1 PID file found."
  exit 0
fi
pid="$(tr -d '[:space:]' < "$PID_FILE")"
if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
  kill -TERM -- "-$pid"
  echo "Stopped strong-interaction curriculum Stage 1 process group $pid."
else
  echo "Stage 1 PID ${pid:-invalid} is not running."
fi
unlink "$PID_FILE"
