#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-}"
case "$PROFILE" in smoke|train) ;; *) echo "Usage: $0 <smoke|train>" >&2; exit 2 ;; esac
PID_FILE="$ROOT/.g11_f_epoch17_student_${PROFILE}.pid"
[[ -f "$PID_FILE" ]] || { echo "G11-F $PROFILE is not running"; exit 0; }
pid="$(tr -d '[:space:]' <"$PID_FILE")"
if ! [[ "$pid" =~ ^[0-9]+$ ]] || ! kill -0 "$pid" 2>/dev/null; then
  unlink "$PID_FILE"; echo "Removed stale G11-F $PROFILE PID file"; exit 0
fi
pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
kill -TERM -- "-$pgid"
echo "Stopped G11-F $PROFILE process group $pgid"
