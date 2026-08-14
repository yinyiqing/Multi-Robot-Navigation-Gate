#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/logs/active/g17-full-scene-comparison"
PID_FILE="$ROOT/.g17_full_scene_comparison.pid"
PIDS=()
cleanup() {
  [[ ${#PIDS[@]} -eq 0 ]] || kill -TERM "${PIDS[@]}" 2>/dev/null || true
  sleep 3
  [[ ${#PIDS[@]} -eq 0 ]] || kill -KILL "${PIDS[@]}" 2>/dev/null || true
  unlink "$PID_FILE" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

run_seed() {
  local seed="$1" status=0 policy port gazebo pid
  local -a methods=(5a r2bbest a1) ports=(17023 17223 17423) gazebos=(17123 17323 17523)
  PIDS=()
  for i in 0 1 2; do
    policy="${methods[$i]}"; port="${ports[$i]}"; gazebo="${gazebos[$i]}"
    bash "$ROOT/scripts/run_g17_full_scene_policy_worker.sh" "$policy" "$seed" "$port" "$gazebo" \
      >"$LOG_DIR/${policy}_s${seed}.runner.log" 2>&1 & PIDS+=("$!")
  done
  for pid in "${PIDS[@]}"; do wait "$pid" || status=1; done
  PIDS=()
  return "$status"
}

mkdir -p "$LOG_DIR"
run_seed 20260824 || exit 1
run_seed 20260825 || exit 1
/usr/bin/python3 "$ROOT/scripts/analyze_g17_full_scene_comparison.py" >"$LOG_DIR/analysis.log" 2>&1 || exit 1
ARCHIVE_DIR="$ROOT/logs/archive/validation/g17_full_scene_comparison"
[[ ! -e "$ARCHIVE_DIR" ]] || { echo "Archive exists: $ARCHIVE_DIR" >&2; exit 1; }
mkdir -p "$(dirname "$ARCHIVE_DIR")"; mv "$LOG_DIR" "$ARCHIVE_DIR"
