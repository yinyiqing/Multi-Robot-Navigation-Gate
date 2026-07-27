#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPEAT="${1:-1}"

[[ "$REPEAT" =~ ^[1-9][0-9]*$ ]] || { echo "Repeat must be a positive integer."; exit 2; }

for label in 5d strong_e16; do
  pid_file="$PROJECT_ROOT/.validation_strong_actor_pair_${label}_r${REPEAT}.pid"
  if [[ ! -f "$pid_file" ]]; then
    echo "No PID file for $label repeat $REPEAT."
    continue
  fi

  pid="$(tr -d '[:space:]' < "$pid_file")"
  if [[ "$pid" =~ ^[0-9]+$ ]]; then
    if kill -- "-$pid" 2>/dev/null; then
      sleep 2
      kill -9 -- "-$pid" 2>/dev/null || true
    elif kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
    echo "Stopped $label repeat $REPEAT process group led by PID $pid."
  else
    echo "Invalid PID file: $pid_file"
  fi
  rm -f "$pid_file"
done
