#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线/26_数量泛化与外部切换基线/local_data/q1"
NUM_AGENTS="${1:?usage: start_g26_q1_evaluation.sh <3|7>}"
[[ "$NUM_AGENTS" == 3 || "$NUM_AGENTS" == 7 ]] || { echo "agent count must be 3 or 7" >&2; exit 2; }
MANIFEST="$BASE/manifests/n${NUM_AGENTS}/test.json"
LAUNCHFILE="$BASE/launch/q1_n${NUM_AGENTS}.launch"
LOG_DIR="$ROOT/logs/active/g26-q1-n${NUM_AGENTS}"
RESULT_DIR="$BASE/results/n${NUM_AGENTS}"
STATE_DIR="$BASE/checkpoints/n${NUM_AGENTS}"
ARCHIVE_DIR="$ROOT/logs/archive/test/g26_q1_n${NUM_AGENTS}"
PID_FILE="$ROOT/.g26_q1_n${NUM_AGENTS}.pid"

[[ -f "$MANIFEST" && -f "$LAUNCHFILE" ]] || { echo "Q1 manifest or launch missing" >&2; exit 1; }
[[ ! -e "$PID_FILE" ]] || { echo "Q1 n${NUM_AGENTS} PID file exists" >&2; exit 1; }
[[ ! -e "$ARCHIVE_DIR" ]] || { echo "Q1 n${NUM_AGENTS} archive already exists" >&2; exit 1; }

mkdir -p "$LOG_DIR" "$RESULT_DIR" "$STATE_DIR"
export Q1_NUM_AGENTS="$NUM_AGENTS" Q1_MANIFEST="$MANIFEST" Q1_EPISODES=128
export Q1_SEEDS="20260911 20260912" Q1_LOG_DIR="$LOG_DIR" Q1_RESULT_DIR="$RESULT_DIR"
export Q1_STATE_DIR="$STATE_DIR" Q1_ARCHIVE_DIR="$ARCHIVE_DIR" Q1_PID_FILE="$PID_FILE"
export Q1_LAUNCHFILE="$LAUNCHFILE" Q1_ROS_PORT=$((18700 + NUM_AGENTS)) Q1_GAZEBO_PORT=$((18730 + NUM_AGENTS))

setsid bash "$ROOT/scripts/run_g26_q1_worker.sh" >>"$LOG_DIR/runner.log" 2>&1 < /dev/null &
echo $! >"$PID_FILE"
echo "Started Q1 ${NUM_AGENTS}-robot evaluation: 2 methods x 128 scenes x 2 repeats"
echo "PID: $(cat "$PID_FILE")"
echo "Live log: $LOG_DIR/runner.log"
