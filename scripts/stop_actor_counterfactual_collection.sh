#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-}"
case "$PROFILE" in
  pilot-train|pilot-validation) ;;
  *) echo "Usage: $0 <pilot-train|pilot-validation>" >&2; exit 2 ;;
esac

PID_FILE="$PROJECT_ROOT/.actor_counterfactual_${PROFILE//-/_}.pid"
if [[ ! -f "$PID_FILE" ]]; then
  echo "No actor counterfactual $PROFILE collection is running."
  exit 0
fi
pid="$(tr -d '[:space:]' < "$PID_FILE")"
if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
fi
unlink "$PID_FILE"
echo "Stopped actor counterfactual $PROFILE collection."
