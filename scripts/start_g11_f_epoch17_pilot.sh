#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATE_ROOT="$ROOT/experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究"
RUN_DIR="$GATE_ROOT/G11_F_epoch17_gate_v1"
LOG_DIR="$ROOT/logs/active/g11_f_epoch17_gate_pilot"
PID_FILE="$ROOT/.g11_f_epoch17_pilot.pid"

declare -A EXPECTED=(
  ["$ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/g11_c_pilot_v1/validation.json.gz"]="1bf044cb5ff9d7d80c14d860d1108481af1d422cf403b26869f8b963012f0e91"
  ["$ROOT/TD3/pytorch_models/TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best_actor.pth"]="fa28855049b67b3ee44c66d55d4f14441fc7c521e5429862c75b152f7d5cacc5"
  ["$ROOT/TD3/pytorch_models/avoidance_actor_from_5a_balanced_continue_e20_s20260813_best_actor.pth"]="149c2e42848ecc9bc478cbed7fd89b9062936dbd5c669b55e6964441685155a5"
  ["$ROOT/experiments/03_保留专门化/02_论文主线/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"]="0b914c0d090bbaba0a2be63c0d75d88580bf4d71778a7f1749c63989e3dbbd56"
  ["$RUN_DIR/local_data/a1_training/seed20260804/any/T1/best.pt"]="b28e81d341c145d6fa8c881dd98c7ece5285231e7d080b3f71afcd2dfe3a0beb"
  ["$RUN_DIR/local_data/aggregated_training/seed20260804/any/T1/best.pt"]="c83a5778d1810213e21af77f681fa9ea30018a9a9d7e75e742ff319d3de58042"
)
for path in "${!EXPECTED[@]}"; do
  [[ -f "$path" ]] || { echo "Missing frozen input: $path" >&2; exit 1; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "${EXPECTED[$path]}" ]] || {
    echo "Frozen input SHA-256 mismatch: $path" >&2; exit 1
  }
done
if [[ -f "$PID_FILE" ]]; then
  pid="$(tr -d '[:space:]' <"$PID_FILE")"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && {
    echo "G11-F pilot already runs as PID $pid" >&2; exit 1
  }
  unlink "$PID_FILE"
fi
if pgrep -af '^python3(\.8)? -u (train|test)_velodyne_td3_multi.py($| )' >/dev/null; then
  echo "Another multi-robot run is active" >&2; exit 1
fi
for port in 16623 16723; do
  ss -ltnH | awk '{print $4}' | grep -Eq ":${port}$" && { echo "Port $port in use" >&2; exit 1; }
done
[[ ! -e "$ROOT/logs/archive/validation/g11_f_epoch17_gate_pilot" ]] || {
  echo "Pilot archive already exists" >&2; exit 1
}
mkdir -p "$LOG_DIR"
setsid bash "$ROOT/scripts/run_g11_f_epoch17_pilot_worker.sh" >>"$LOG_DIR/runner.log" 2>&1 < /dev/null &
echo $! >"$PID_FILE"
echo "Started G11-F epoch-17 fixed closed-loop pilot"
echo "PID: $(cat "$PID_FILE")"
echo "Runs: 5A/A1/B2 x 2 repeats = 300 episodes"
echo "Runner: $LOG_DIR/runner.log"
