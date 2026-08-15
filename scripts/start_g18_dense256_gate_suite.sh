#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
LOG_DIR="$ROOT/logs/active/g18-dense256-gate-suite"
PID_FILE="$ROOT/.g18_dense256_gate_suite.pid"

verify_sha() {
  local path="$1" expected="$2"
  [[ -f "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]] || {
    echo "SHA-256 mismatch: $path" >&2; exit 1
  }
}

verify_sha "$BASE/datasets/fixed_v1/dense/validation.json.gz" "2d1dde389f927b924fa5993c47460bc60bac42aa9506ae3869c3139c9d1264b7"
verify_sha "$ROOT/TD3/pytorch_models/TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best_actor.pth" "fa28855049b67b3ee44c66d55d4f14441fc7c521e5429862c75b152f7d5cacc5"
verify_sha "$ROOT/TD3/pytorch_models/avoidance_actor_from_5a_balanced_continue_e20_s20260813_best_actor.pth" "149c2e42848ecc9bc478cbed7fd89b9062936dbd5c669b55e6964441685155a5"
verify_sha "$BASE/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt" "0b914c0d090bbaba0a2be63c0d75d88580bf4d71778a7f1749c63989e3dbbd56"
verify_sha "$BASE/11_可部署在线Gate研究/G11_F_epoch17_gate_v1/local_data/a1_training/seed20260804/any/T1/best.pt" "b28e81d341c145d6fa8c881dd98c7ece5285231e7d080b3f71afcd2dfe3a0beb"
verify_sha "$BASE/11_可部署在线Gate研究/G11_F_epoch17_gate_v1/local_data/aggregated_training/seed20260804/any/T1/best.pt" "c83a5778d1810213e21af77f681fa9ea30018a9a9d7e75e742ff319d3de58042"
verify_sha "$BASE/11_可部署在线Gate研究/G11_B_student_rollout_v1/local_data/training/seed20260804/any/T1/best.pt" "fc59b4f783f7c5461ebb0239fab4b34896ad910ee78e7223e88d29ce9c3f5a52"

if [[ -f "$PID_FILE" ]]; then
  pid="$(tr -d '[:space:]' <"$PID_FILE")"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && {
    echo "G18 Gate suite already queued or running as PID $pid" >&2; exit 1
  }
  unlink "$PID_FILE"
fi
[[ ! -e "$ROOT/logs/archive/validation/g18_dense256_gate_suite" ]] || {
  echo "G18 Gate suite archive already exists" >&2; exit 1
}

mkdir -p "$LOG_DIR"
/usr/bin/python3 "$ROOT/scripts/generate_multi_robot_launch.py" --num-agents 5 \
  --output "$LOG_DIR/runtime_g18_dense256_gate_suite.launch"
setsid bash "$ROOT/scripts/run_g18_dense256_gate_suite_worker.sh" >>"$LOG_DIR/runner.log" 2>&1 < /dev/null &
echo $! >"$PID_FILE"
echo "G18 dense256 Gate suite queued"
echo "PID: $(cat "$PID_FILE")"
echo "Log: $LOG_DIR/runner.log"
