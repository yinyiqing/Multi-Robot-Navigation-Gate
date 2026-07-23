#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$PROJECT_ROOT/.train_strong_interaction_expert_pilot.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No strong-interaction pilot PID file found."
  exit 0
fi
pid="$(tr -d '[:space:]' < "$PID_FILE")"
if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
  kill -TERM -- "-$pid"
  echo "Stopped strong-interaction pilot process group $pid."
else
  echo "Strong-interaction pilot PID ${pid:-invalid} is not running."
fi
rm -f "$PID_FILE"
