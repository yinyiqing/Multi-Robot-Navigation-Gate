#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
RUN="$BASE/37_最终双监督消融"
PID_FILE="$ROOT/.g34_phase_only_ablation.pid"
QUEUE_LOG="$RUN/logs/matched/runner.log"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' <"$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "G34 phase-only ablation is already running as PID $old_pid"
    exit 0
  fi
  unlink "$PID_FILE"
fi

mkdir -p "$RUN/logs/matched"
source "$ROOT/env.python.sh"
python3 "$ROOT/scripts/prepare_g34_phase_only_checkpoint.py" \
  >"$RUN/logs/matched/checkpoint_prepare.log" 2>&1
git -C "$ROOT" rev-parse HEAD >"$RUN/logs/matched/git_commit.txt"
sha256sum "$ROOT/TD3/learned_gate_controller.py" "$ROOT/TD3/reward_aware_gate.py" \
  "$ROOT/TD3/test_velodyne_td3_multi.py" "$ROOT/TD3/multi_agent_velodyne_env.py" \
  "$ROOT/scripts/run_g34_g25slice_matched_worker.sh" \
  "$ROOT/scripts/run_g34_phase_only_ablation_queue.sh" \
  "$ROOT/scripts/analyze_g34_phase_only_ablation.py" \
  >"$RUN/logs/matched/code_sha256.txt"

setsid bash "$ROOT/scripts/run_g34_phase_only_ablation_queue.sh" \
  >>"$QUEUE_LOG" 2>&1 < /dev/null &
pid=$!
echo "$pid" >"$PID_FILE"
echo "Started G34 phase-only ablation queue as PID $pid"
echo "Live log: $QUEUE_LOG"
