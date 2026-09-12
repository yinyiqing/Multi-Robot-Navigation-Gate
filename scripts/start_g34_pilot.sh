#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
G34="$ROOT/experiments/03_保留专门化/02_论文主线/34_双头RewardAwareGate"
PID_FILE="$ROOT/.g34_pilot.pid"
QUEUE_LOG="$G34/logs/pilot32/queue.log"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' <"$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "G34 pilot is already running as PID $old_pid"
    exit 0
  fi
fi

mkdir -p "$(dirname "$QUEUE_LOG")"
nohup setsid bash "$ROOT/scripts/run_g34_pilot.sh" >"$QUEUE_LOG" 2>&1 </dev/null &
pid=$!
echo "Started G34 Dense32 pilot queue as PID $pid"
echo "Log: $QUEUE_LOG"
