#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
LOG_DIR="$ROOT/logs/active/g18-dense256-r2b"
PID_FILE="$ROOT/.g18_dense256_r2b.pid"

verify_sha() {
  local path="$1" expected="$2"
  [[ -f "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]] || {
    echo "SHA-256 mismatch: $path" >&2; exit 1
  }
}

verify_sha "$BASE/datasets/fixed_v1/dense/validation.json.gz" \
  "2d1dde389f927b924fa5993c47460bc60bac42aa9506ae3869c3139c9d1264b7"
verify_sha "$ROOT/TD3/pytorch_models/capacity_wide_r2b_5a_recipe_n5_seed20260823_best_actor.pth" \
  "da28dd5820d09845eea07cb68da45a7afd262fe56e8a71f80bf6b5781551523a"

if [[ -f "$PID_FILE" ]]; then
  pid="$(tr -d '[:space:]' <"$PID_FILE")"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && {
    echo "G18 already queued or running as PID $pid" >&2; exit 1
  }
  unlink "$PID_FILE"
fi
[[ ! -e "$ROOT/logs/archive/validation/g18_dense256_r2b" ]] || {
  echo "G18 archive already exists" >&2; exit 1
}

mkdir -p "$LOG_DIR"
/usr/bin/python3 "$ROOT/scripts/generate_multi_robot_launch.py" --num-agents 5 \
  --output "$LOG_DIR/runtime_g18_dense256.launch"
setsid bash "$ROOT/scripts/run_g18_dense256_r2b_worker.sh" >>"$LOG_DIR/runner.log" 2>&1 < /dev/null &
echo $! >"$PID_FILE"
echo "G18 dense256 R2B-best queued"
echo "PID: $(cat "$PID_FILE")"
echo "Log: $LOG_DIR/runner.log"
