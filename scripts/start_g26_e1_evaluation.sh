#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
E1="$BASE/26_数量泛化与外部切换基线/local_data/e1"
MANIFEST="$E1/manifests/dense_test_256_384.json.gz"
RECORD="$E1/manifests/manifest_record.json"
CHECKPOINT="$E1/nf_switch_seed20260821/checkpoint.pt"
SUMMARY="$E1/nf_switch_seed20260821/summary.json"
LOG_DIR="$ROOT/logs/active/g26-e1-evaluation"
ARCHIVE_DIR="$ROOT/logs/archive/test/g26_e1_evaluation"
PID_FILE="$ROOT/.g26_e1_evaluation.pid"

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
verify_sha "$BASE/11_可部署在线Gate研究/G11_B_student_rollout_v1/local_data/training/seed20260804/any/T1/best.pt" "fc59b4f783f7c5461ebb0239fab4b34896ad910ee78e7223e88d29ce9c3f5a52"
verify_sha "$CHECKPOINT" "bf43581ba0aab37f96f267f094f858143005b5876e0c5d4a31cd79c1eedeb6af"
verify_sha "$SUMMARY" "c465058e9d9a6e78b6fafb230a574368c9a3a426d10af0126935b531fb6eea45"

python3 - "$MANIFEST" "$RECORD" <<'PY'
import gzip
import hashlib
import json
import sys
from pathlib import Path

manifest_path, record_path = map(Path, sys.argv[1:])
if not manifest_path.is_file() or not record_path.is_file():
    raise SystemExit("E1 manifest and record are required before evaluation")
record = json.loads(record_path.read_text(encoding="utf-8"))
with gzip.open(manifest_path, "rt", encoding="utf-8") as handle:
    payload = json.load(handle)
scenarios = payload.get("scenarios", [])
if len(scenarios) != 128:
    raise SystemExit("E1 manifest must contain exactly 128 scenes")
if any(int(item.get("num_agents", -1)) != 5 for item in scenarios):
    raise SystemExit("E1 manifest contains a non-five-robot scene")
if record.get("scenes") != 128:
    raise SystemExit("E1 manifest record has the wrong scene count")
actual = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
if record.get("output_sha256") != actual:
    raise SystemExit("E1 manifest record hash mismatch")
if record.get("start_index_in_source") != 256 or record.get("stop_index_in_source") != 384:
    raise SystemExit("E1 manifest record has the wrong source slice")
if record.get("sealed_test_read") is not True:
    raise SystemExit("E1 manifest record must mark the test slice as read")
PY

[[ ! -e "$PID_FILE" ]] || { echo "E1 PID file exists: $PID_FILE" >&2; exit 1; }
[[ ! -e "$ARCHIVE_DIR" ]] || { echo "E1 archive already exists: $ARCHIVE_DIR" >&2; exit 1; }

mkdir -p "$LOG_DIR" "$E1/results" "$E1/checkpoints"
/usr/bin/python3 "$ROOT/scripts/generate_multi_robot_launch.py" --num-agents 5 --output "$LOG_DIR/runtime_e1.launch"
git -C "$ROOT" rev-parse HEAD >"$LOG_DIR/git_commit.txt"
export G26_E1_MANIFEST="$MANIFEST" G26_E1_EPISODES=128
export G26_E1_SEEDS="20260921 20260922"
export G26_E1_LOG_DIR="$LOG_DIR" G26_E1_ARCHIVE_DIR="$ARCHIVE_DIR"
export G26_E1_RESULT_DIR="$E1/results" G26_E1_STATE_DIR="$E1/checkpoints"
export G26_E1_PID_FILE="$PID_FILE" G26_E1_LAUNCHFILE="$LOG_DIR/runtime_e1.launch"
export G26_E1_ROS_PORT=18623 G26_E1_GAZEBO_PORT=18723
export G26_E1_NF_CHECKPOINT="$CHECKPOINT"

setsid bash "$ROOT/scripts/run_g26_e1_worker.sh" >>"$LOG_DIR/runner.log" 2>&1 < /dev/null &
echo $! >"$PID_FILE"
echo "Started G26-E1 evaluation: 3 methods x 128 scenes x 2 repeats"
echo "PID: $(cat "$PID_FILE")"
echo "Live log: $LOG_DIR/runner.log"
