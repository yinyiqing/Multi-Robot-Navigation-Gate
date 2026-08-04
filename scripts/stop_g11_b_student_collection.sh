#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-}"
case "$PROFILE" in
  smoke|train) ;;
  *) echo "Usage: $0 <smoke|train>" >&2; exit 2 ;;
esac

PID_FILE="$PROJECT_ROOT/.g11_b_student_${PROFILE}.pid"
if [[ ! -f "$PID_FILE" ]]; then
  echo "G11-B $PROFILE collection is not running."
  exit 0
fi
pid="$(tr -d '[:space:]' < "$PID_FILE")"
if ! [[ "$pid" =~ ^[0-9]+$ ]] || ! kill -0 "$pid" 2>/dev/null; then
  unlink "$PID_FILE"
  echo "Removed stale G11-B $PROFILE PID file."
  exit 0
fi
pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
kill -TERM -- "-$pgid"
echo "Stopped G11-B $PROFILE process group $pgid."
