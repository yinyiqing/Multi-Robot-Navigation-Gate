#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
G27="$BASE/27_反事实Reward增强Gate监督"
MANIFEST="$BASE/datasets/fixed_v1/dense/validation.json.gz"
TRAINING="$G27/local_data/training/seed20260910"
CHECKPOINT="$TRAINING/best.pt"
SUMMARY="$TRAINING/summary.json"
LOG_DIR="$G27/logs/p3_validation"
PID_FILE="$ROOT/.g27_p3_validation.pid"

[[ -f "$SUMMARY" && -f "$CHECKPOINT" ]] || { echo "G27 B3 training is incomplete" >&2; exit 2; }
python3 - "$SUMMARY" "$CHECKPOINT" <<'PY'
import hashlib,json,sys
summary=json.load(open(sys.argv[1],encoding='utf-8'))
actual=hashlib.sha256(open(sys.argv[2],'rb').read()).hexdigest()
if summary.get('checkpoint',{}).get('sha256') != actual:
    raise SystemExit('B3 checkpoint/summary hash mismatch')
PY
[[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" == "2d1dde389f927b924fa5993c47460bc60bac42aa9506ae3869c3139c9d1264b7" ]] || {
  echo "Dense validation manifest hash mismatch" >&2; exit 1;
}
if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' <"$PID_FILE")"
  [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null && {
    echo "G27 P3 already running as PID $old_pid" >&2; exit 1;
  }
  unlink "$PID_FILE"
fi

mkdir -p "$LOG_DIR"
/usr/bin/python3 "$ROOT/scripts/generate_multi_robot_launch.py" --num-agents 5 \
  --output "$LOG_DIR/runtime_g27_p3.launch"
setsid bash "$ROOT/scripts/run_g27_p3_validation_worker.sh" \
  >"$LOG_DIR/runner.log" 2>&1 < /dev/null &
echo $! >"$PID_FILE"
echo "Started G27 P3 B3-only Dense256 validation."
echo "PID: $(cat "$PID_FILE")"
echo "Log: $LOG_DIR/runner.log"
