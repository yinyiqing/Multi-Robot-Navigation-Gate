#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
RUN_DIR="$BASE/17_完整场景统一对比"
LOG_DIR="$ROOT/logs/active/g17-full-scene-comparison"
PID_FILE="$ROOT/.g17_full_scene_comparison.pid"

verify_sha() {
  local path="$1" expected="$2"
  [[ -f "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]] || {
    echo "SHA-256 mismatch: $path" >&2; exit 1
  }
}

verify_sha "$BASE/datasets/fixed_v1/views/g12_full_scene_selection_v1/validation.json.gz" "52435d6c5bdf9914e7212dd29cb4bfec074257f72d85f0d71741deee7c63b635"
verify_sha "$ROOT/TD3/pytorch_models/TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best_actor.pth" "fa28855049b67b3ee44c66d55d4f14441fc7c521e5429862c75b152f7d5cacc5"
verify_sha "$ROOT/TD3/pytorch_models/capacity_wide_r2b_5a_recipe_n5_seed20260823_best_actor.pth" "da28dd5820d09845eea07cb68da45a7afd262fe56e8a71f80bf6b5781551523a"
verify_sha "$ROOT/TD3/pytorch_models/avoidance_actor_from_5a_balanced_continue_e20_s20260813_best_actor.pth" "149c2e42848ecc9bc478cbed7fd89b9062936dbd5c669b55e6964441685155a5"
verify_sha "$BASE/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt" "0b914c0d090bbaba0a2be63c0d75d88580bf4d71778a7f1749c63989e3dbbd56"
verify_sha "$BASE/11_可部署在线Gate研究/G11_F_epoch17_gate_v1/local_data/a1_training/seed20260804/any/T1/best.pt" "b28e81d341c145d6fa8c881dd98c7ece5285231e7d080b3f71afcd2dfe3a0beb"

if [[ -f "$PID_FILE" ]]; then
  pid="$(tr -d '[:space:]' <"$PID_FILE")"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && {
    echo "G17 already runs as PID $pid" >&2; exit 1
  }
  unlink "$PID_FILE"
fi
pgrep -af '^python3(\.8)? -u (train|test)_velodyne_td3_multi.py($| )' >/dev/null && {
  echo "Another multi-robot run is active" >&2; exit 1
}
[[ ! -e "$ROOT/logs/archive/validation/g17_full_scene_comparison" ]] || {
  echo "G17 archive already exists" >&2; exit 1
}
for port in 17023 17123; do
  ss -ltnH | awk '{print $4}' | grep -Eq ":${port}$" && { echo "Port $port in use" >&2; exit 1; }
done

mkdir -p "$LOG_DIR" "$RUN_DIR/local_data"
/usr/bin/python3 "$ROOT/scripts/generate_multi_robot_launch.py" --num-agents 5 \
  --output "$LOG_DIR/runtime_g17_full_scene.launch"
setsid bash "$ROOT/scripts/run_g17_full_scene_comparison_worker.sh" >>"$LOG_DIR/runner.log" 2>&1 < /dev/null &
echo $! >"$PID_FILE"
echo "G17 full-scene comparison started"
echo "PID: $(cat "$PID_FILE")"
echo "Episodes: 720 total"
echo "Log: $LOG_DIR/runner.log"
