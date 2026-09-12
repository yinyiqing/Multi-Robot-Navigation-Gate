#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
G28="$ROOT/experiments/03_保留专门化/02_论文主线/28_RewardAwareGate稳健修正版"
B4_RESULT="$G28/local_data/validation/results/g28_dense256_b4_s20260911.npy"
MANIFEST="$ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/dense/validation.json.gz"
LOG="$G28/logs/b2_after_b4.log"

verify_b4() {
  python3 - "$B4_RESULT" "$MANIFEST" <<'PY'
import gzip, json, sys, numpy as np
rows = np.load(sys.argv[1], allow_pickle=True)
with gzip.open(sys.argv[2], "rt", encoding="utf-8") as handle:
    expected = [str(item["scenario_id"]) for item in json.load(handle)["scenarios"][:256]]
if rows.shape != (256, 17): raise SystemExit(1)
if [str(item) for item in rows[:, 12]] != expected: raise SystemExit(1)
if sum(int(row[6]) + int(row[7]) + int(row[10]) for row in rows) != 1280: raise SystemExit(1)
PY
}

while ! verify_b4 2>/dev/null; do
  if ! pgrep -f '[t]est_velodyne_td3_multi.py' >/dev/null; then
    echo "B4 process ended before a complete result; B2 control not started" >&2
    exit 1
  fi
  sleep 30
done

echo "B4 result verified; starting same-seed frozen B2 control" | tee -a "$LOG"
G28_P3_METHOD=b2 G28_P3_SEED=20260911 bash "$ROOT/scripts/run_g28_p3_validation.sh" >>"$LOG" 2>&1
