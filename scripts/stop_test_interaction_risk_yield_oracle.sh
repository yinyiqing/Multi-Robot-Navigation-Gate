#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$PROJECT_ROOT/.test_interaction_risk_yield_oracle.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "Interaction-risk yield-oracle PID file is absent."
  exit 0
fi

pid="$(cat "$PID_FILE")"
if kill -0 "$pid" 2>/dev/null; then
  pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
  kill -- "-$pgid" 2>/dev/null || kill "$pid" 2>/dev/null || true
fi
rm -f "$PID_FILE"
echo "Stopped interaction-risk yield oracle."
