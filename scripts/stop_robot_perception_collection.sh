#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPLIT="${1:-}"
case "$SPLIT" in
  train|validation) ;;
  *) echo "Usage: $0 <train|validation>" >&2; exit 2 ;;
esac
PID_FILE="$PROJECT_ROOT/.robot_perception_collection_${SPLIT}.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No managed robot-perception $SPLIT collection is running."
  exit 0
fi

pid="$(tr -d '[:space:]' < "$PID_FILE")"
if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
  pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
  kill -- "-$pgid" 2>/dev/null || kill "$pid" 2>/dev/null || true
  echo "Stopped robot-perception $SPLIT process group $pgid."
else
  echo "Managed robot-perception $SPLIT collection is not active."
fi
unlink "$PID_FILE"
