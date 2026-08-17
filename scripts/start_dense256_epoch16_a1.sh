#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
LOG_DIR="$ROOT/logs/active/dense256-epoch16-a1"
PID_FILE="$ROOT/.dense256_epoch16_a1.pid"
verify_sha() {
  local path="$1" expected="$2"
  [[ -f "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]] || { echo "SHA mismatch: $path" >&2; exit 1; }
}
verify_sha "$BASE/datasets/fixed_v1/dense/validation.json.gz" "2d1dde389f927b924fa5993c47460bc60bac42aa9506ae3869c3139c9d1264b7"
verify_sha "$ROOT/TD3/pytorch_models/TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best_actor.pth" "fa28855049b67b3ee44c66d55d4f14441fc7c521e5429862c75b152f7d5cacc5"
verify_sha "$ROOT/TD3/pytorch_models/interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016_actor.pth" "6ec1942fcd497ab1cc2a85a5aaec8f524395dc21ff21a442dca243a52e917c0b"
verify_sha "$BASE/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt" "0b914c0d090bbaba0a2be63c0d75d88580bf4d71778a7f1749c63989e3dbbd56"
verify_sha "$BASE/11_可部署在线Gate研究/G11_A1_当前协议时序pilot/local_data/training/seed20260804/any/T1/best.pt" "d9b05d9f86e5bad4d2071c041187b618ebca6f1a3cc1f9c46e8b14b1a451537a"
mkdir -p "$LOG_DIR"
if [[ -f "$PID_FILE" ]]; then
  pid="$(tr -d '[:space:]' <"$PID_FILE")"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && { echo "Already running: $pid"; exit 0; }
  unlink "$PID_FILE"
fi
setsid bash "$ROOT/scripts/run_dense256_epoch16_a1_worker.sh" >>"$LOG_DIR/runner.log" 2>&1 < /dev/null &
echo $! >"$PID_FILE"
echo "Started Dense256 epoch16+A1."
echo "PID: $(cat "$PID_FILE")"
echo "Live log: $LOG_DIR/runner.log"
