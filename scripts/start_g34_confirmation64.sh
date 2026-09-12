#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
G34="$BASE/34_双头RewardAwareGate"
PID_FILE="$ROOT/.g34_confirmation64.pid"
QUEUE_LOG="$G34/logs/confirmation64/queue.log"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' <"$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "G34 confirmation64 is already running as PID $old_pid"
    exit 0
  fi
fi

mkdir -p "$(dirname "$QUEUE_LOG")"
export G34_STAGE=confirmation64
export G34_RUN_NAME=g34_dense64_confirmation
export G34_MANIFEST="$G34/local_data/protocol/dense_confirmation64.json.gz"
export G34_MANIFEST_SHA256=4f70d1f7536f9619c5f687ea98c24d3a68b985e9a1d3673066d5df8c1f5ed4f7
export G34_SEED=20260913
export G34_TARGET_EPISODES=64
export G34_PID_FILE="$PID_FILE"
export G34_ROS_PORT=18641
export G34_GAZEBO_PORT=19641
export G34_ANALYZER="$ROOT/scripts/analyze_g34_confirmation64.py"
nohup setsid bash "$ROOT/scripts/run_g34_pilot.sh" >"$QUEUE_LOG" 2>&1 </dev/null &
pid=$!
echo "Started G34 Dense64 confirmation as PID $pid"
echo "Log: $QUEUE_LOG"
