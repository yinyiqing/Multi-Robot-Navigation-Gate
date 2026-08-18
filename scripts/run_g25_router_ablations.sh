#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
RUN_DIR="$BASE/25_最终消融与Sealed评测/local_data/router_ablations"
LOG_DIR="$ROOT/logs/active/g25-router-ablations"
ARCHIVE_DIR="$ROOT/logs/archive/training/g25_router_ablations"
SEED=20260804

verify_sha() {
  local path="$1" expected="$2"
  [[ -f "$path" ]] || { echo "Missing frozen input: $path" >&2; exit 1; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]] || {
    echo "SHA-256 mismatch: $path" >&2
    exit 1
  }
}

verify_sha "$ROOT/TD3/pytorch_models/TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best_actor.pth" "fa28855049b67b3ee44c66d55d4f14441fc7c521e5429862c75b152f7d5cacc5"
verify_sha "$ROOT/TD3/pytorch_models/interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016_actor.pth" "6ec1942fcd497ab1cc2a85a5aaec8f524395dc21ff21a442dca243a52e917c0b"
verify_sha "$BASE/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt" "0b914c0d090bbaba0a2be63c0d75d88580bf4d71778a7f1749c63989e3dbbd56"
mkdir -p "$RUN_DIR" "$LOG_DIR"
set +u
source "$ROOT/env.python.sh"
set -u
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

run_one() {
  local id="$1" sequence_length="$2" feature_set="$3"
  local output="$RUN_DIR/$id" log="$LOG_DIR/train_${id}.log"
  if [[ -f "$output/summary.json" ]]; then
    echo "Refusing to overwrite completed $id" >&2
    return 1
  fi
  mkdir -p "$output"
  echo "Starting $id: sequence_length=$sequence_length feature_set=$feature_set"
  /usr/bin/time -f 'elapsed=%E maxrss_kb=%M' nice -n 15 python3 \
    "$ROOT/scripts/train_g11_b_aggregated_gate.py" \
    --experiment-id "G25-$id" \
    --epochs 40 \
    --seed "$SEED" \
    --device cpu \
    --sequence-length "$sequence_length" \
    --feature-set "$feature_set" \
    --output-dir "$output" \
    >"$log" 2>&1
  echo "Completed $id"
}

run_one v4_single_frame 1 base_and_actor_actions
run_one v5_no_action_difference 8 base

python3 - "$RUN_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
expected = {
    "v4_single_frame": (1, "base_and_actor_actions", 82),
    "v5_no_action_difference": (8, "base", 76),
}
summary = {}
for name, (sequence_length, feature_set, input_dim) in expected.items():
    checkpoint = root / name / "any/T1/best.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload["sequence_length"]) != sequence_length:
        raise SystemExit("%s sequence length mismatch" % name)
    if payload["feature_set"] != feature_set:
        raise SystemExit("%s feature set mismatch" % name)
    if int(payload["model_config"]["input_dim"]) != input_dim:
        raise SystemExit("%s input dimension mismatch" % name)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    metrics = json.loads((root / name / "summary.json").read_text())["result"]
    summary[name] = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "sequence_length": sequence_length,
        "feature_set": feature_set,
        "input_dim": input_dim,
        "threshold": metrics["threshold"],
        "best_epoch": metrics["best_epoch"],
        "parameter_count": metrics["parameter_count"],
        "validation": metrics["validation"],
    }
(root / "audit.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

[[ ! -e "$ARCHIVE_DIR" ]] || { echo "Archive already exists: $ARCHIVE_DIR" >&2; exit 1; }
mkdir -p "$(dirname "$ARCHIVE_DIR")"
mv "$LOG_DIR" "$ARCHIVE_DIR"
echo "G25 Router ablation training complete. Logs: $ARCHIVE_DIR"
