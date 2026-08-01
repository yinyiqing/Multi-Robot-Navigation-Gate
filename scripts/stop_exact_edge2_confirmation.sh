#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-}"
case "$MODE" in
  5a|learned-gate) ;;
  *) echo "Usage: $0 <5a|learned-gate>" >&2; exit 2 ;;
esac

PID_FILE="$PROJECT_ROOT/.edge2_confirmation_${MODE//-/_}.pid"
if [[ ! -f "$PID_FILE" ]]; then
  echo "No exact-edge-2 $MODE confirmation is running."
  exit 0
fi
pid="$(tr -d '[:space:]' < "$PID_FILE")"
if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
fi
unlink "$PID_FILE"
echo "Stopped exact-edge-2 $MODE confirmation."
