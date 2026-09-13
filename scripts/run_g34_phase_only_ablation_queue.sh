#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
RUN="$BASE/37_最终双监督消融"
MATCHED="$RUN/local_data/matched"
MANIFEST="$BASE/25_最终消融与Sealed评测/local_data/sealed_manifest/dense_test_first256.json.gz"
CHECKPOINT="$RUN/local_data/checkpoint/g34_phase_only.pt"
OUTER_PID_FILE="$ROOT/.g34_phase_only_ablation.pid"
WORKER_PID_FILE="$ROOT/.g34_phase_only_ablation_worker.pid"
LAUNCHFILE="$RUN/logs/matched/runtime_g34_phase_only.launch"

cleanup() { unlink "$OUTER_PID_FILE" "$WORKER_PID_FILE" 2>/dev/null || true; }
trap cleanup EXIT

export G34_MATCHED_RUN="$RUN"
export G34_MATCHED_MANIFEST="$MANIFEST"
export G34_MATCHED_LOG_DIR="$RUN/logs/matched"
export G34_MATCHED_RESULT_DIR="$MATCHED/results"
export G34_MATCHED_STATE_DIR="$MATCHED/checkpoints"
export G34_MATCHED_PID_FILE="$WORKER_PID_FILE"
export G34_MATCHED_LAUNCHFILE="$LAUNCHFILE"
export G34_MATCHED_CHECKPOINT="$CHECKPOINT"
export G34_MATCHED_RUN_PREFIX="g34_phase_only"
export G34_MATCHED_EXPERIMENT_ID="G34-final-phase-only-ablation"
export G34_MATCHED_METHOD_ID="phase_only"
export G34_MATCHED_COMPLETION_PATH="$MATCHED/completion.json"
export G34_MATCHED_ROS_PORT=18653
export G34_MATCHED_GAZEBO_PORT=19653

mkdir -p "$RUN/logs/matched" "$MATCHED/results" "$MATCHED/checkpoints"
/usr/bin/python3 "$ROOT/scripts/generate_multi_robot_launch.py" --num-agents 5 --output "$LAUNCHFILE"

echo "Starting 16-scene infrastructure pilot"
export G34_MATCHED_EPISODES=16
export G34_MATCHED_SEEDS=20260901
setsid bash "$ROOT/scripts/run_g34_g25slice_matched_worker.sh"
cp "$MATCHED/results/g34_phase_only_s20260901.npy" "$MATCHED/pilot_result_s20260901.npy"
cp "$RUN/logs/matched/g34_phase_only_s20260901_attempt1.log" "$RUN/logs/matched/pilot_attempt1.log"

python3 - "$MATCHED/results/g34_phase_only_s20260901.npy" "$MATCHED/pilot_audit.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
import numpy as np
source,output=Path(sys.argv[1]),Path(sys.argv[2])
rows=np.load(source,allow_pickle=True)
if rows.shape != (16,17): raise SystemExit('phase-only pilot shape mismatch')
record={
    'format_version':1,
    'status':'pass',
    'purpose':'infrastructure and terminal-accounting audit only',
    'episodes':16,
    'seed':20260901,
    'result_prefix_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'performance_used_as_stop_condition':False,
}
output.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
PY

echo "Pilot passed; resuming the same first repeat and running all three repeats"
export G34_MATCHED_EPISODES=256
export G34_MATCHED_SEEDS="20260901 20260902 20260903"
setsid bash "$ROOT/scripts/run_g34_g25slice_matched_worker.sh"

source "$ROOT/env.python.sh"
python3 "$ROOT/scripts/analyze_g34_phase_only_ablation.py" >"$RUN/logs/matched/statistics.log" 2>&1
echo "G34 final phase-only ablation complete"
