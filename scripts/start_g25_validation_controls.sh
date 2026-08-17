#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
LOG_DIR="$ROOT/logs/active/g25-validation-controls"
ARCHIVE_DIR="$ROOT/logs/archive/validation/g25_validation_controls"
PID_FILE="$ROOT/.g25_validation_controls.pid"

verify_sha() {
  local path="$1" expected="$2"
  [[ -f "$path" ]] || { echo "Missing frozen input: $path" >&2; exit 1; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]] || {
    echo "SHA-256 mismatch: $path" >&2
    exit 1
  }
}

verify_sha "$BASE/datasets/fixed_v1/dense/validation.json.gz" "2d1dde389f927b924fa5993c47460bc60bac42aa9506ae3869c3139c9d1264b7"
verify_sha "$ROOT/TD3/pytorch_models/TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best_actor.pth" "fa28855049b67b3ee44c66d55d4f14441fc7c521e5429862c75b152f7d5cacc5"
verify_sha "$ROOT/TD3/pytorch_models/interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016_actor.pth" "6ec1942fcd497ab1cc2a85a5aaec8f524395dc21ff21a442dca243a52e917c0b"
verify_sha "$BASE/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt" "0b914c0d090bbaba0a2be63c0d75d88580bf4d71778a7f1749c63989e3dbbd56"
verify_sha "$BASE/11_可部署在线Gate研究/G11_B_student_rollout_v1/local_data/training/seed20260804/any/T1/best.pt" "fc59b4f783f7c5461ebb0239fab4b34896ad910ee78e7223e88d29ce9c3f5a52"

if [[ -f "$PID_FILE" ]]; then
  pid="$(tr -d '[:space:]' <"$PID_FILE")"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && {
    echo "G25 validation controls already queued or running as PID $pid" >&2
    exit 1
  }
  unlink "$PID_FILE"
fi
[[ ! -e "$ARCHIVE_DIR" ]] || { echo "G25 archive already exists: $ARCHIVE_DIR" >&2; exit 1; }

mkdir -p "$LOG_DIR"
/usr/bin/python3 "$ROOT/scripts/generate_multi_robot_launch.py" --num-agents 5 \
  --output "$LOG_DIR/runtime_g25_validation_controls.launch"
setsid bash "$ROOT/scripts/run_g25_validation_controls_worker.sh" >>"$LOG_DIR/runner.log" 2>&1 < /dev/null &
echo $! >"$PID_FILE"

echo "G25 validation controls queued (CPU-only, serial)."
echo "PID: $(cat "$PID_FILE")"
echo "Runs: min-LiDAR rule, then B2 without hysteresis/hold"
echo "Live log: $LOG_DIR/runner.log"
