#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT/experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究/G11_F_epoch17_gate_v1"
LOG_DIR="$ROOT/logs/active/g11_f_epoch17_gate_r2b30k_pilot"
PID_FILE="$ROOT/.g11_f_epoch17_r2b30k_pilot.pid"

declare -A EXPECTED=(
  ["$ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/g11_c_pilot_v1/validation.json.gz"]="1bf044cb5ff9d7d80c14d860d1108481af1d422cf403b26869f8b963012f0e91"
  ["$ROOT/TD3/pytorch_models/capacity_wide_r2b_5a_recipe_n5_seed20260823_epoch_003_actor.pth"]="2722574e531080d9ce4f2448687270354a8baea0563c12c78e9711a8da4f62f2"
  ["$RUN_DIR/local_data/pilot/results/g11_f_c_5a_r1_s20260805.npy"]=""
  ["$RUN_DIR/local_data/pilot/results/g11_f_c_a1_r1_s20260805.npy"]=""
  ["$RUN_DIR/local_data/pilot/results/g11_f_c_b2_r1_s20260805.npy"]=""
  ["$RUN_DIR/local_data/pilot/results/g11_f_c_5a_r2_s20260806.npy"]=""
  ["$RUN_DIR/local_data/pilot/results/g11_f_c_a1_r2_s20260806.npy"]=""
  ["$RUN_DIR/local_data/pilot/results/g11_f_c_b2_r2_s20260806.npy"]=""
)
for path in "${!EXPECTED[@]}"; do
  [[ -f "$path" ]] || { echo "Missing frozen input: $path" >&2; exit 1; }
  if [[ -n "${EXPECTED[$path]}" ]]; then
    [[ "$(sha256sum "$path" | awk '{print $1}')" == "${EXPECTED[$path]}" ]] || {
      echo "Frozen input SHA-256 mismatch: $path" >&2; exit 1
    }
  fi
done
if [[ -f "$PID_FILE" ]]; then
  pid="$(tr -d '[:space:]' <"$PID_FILE")"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && {
    echo "G11-F R2B-30k pilot already runs as PID $pid" >&2; exit 1
  }
  unlink "$PID_FILE"
fi
if pgrep -af '^python3(\.8)? -u (train|test)_velodyne_td3_multi.py($| )' >/dev/null; then
  echo "Another multi-robot run is active" >&2; exit 1
fi
for port in 16823 16923; do
  ss -ltnH | awk '{print $4}' | grep -Eq ":${port}$" && { echo "Port $port in use" >&2; exit 1; }
done
[[ ! -e "$ROOT/logs/archive/validation/g11_f_epoch17_gate_r2b30k_pilot" ]] || {
  echo "R2B-30k pilot archive already exists" >&2; exit 1
}
mkdir -p "$LOG_DIR"
setsid bash "$ROOT/scripts/run_g11_f_epoch17_r2b30k_pilot_worker.sh" >>"$LOG_DIR/runner.log" 2>&1 < /dev/null &
echo $! >"$PID_FILE"
echo "Started G11-F R2B-30k fair capacity extension"
echo "PID: $(cat "$PID_FILE")"
echo "Runs: R2B-30k x 2 repeats = 100 episodes"
echo "Runner: $LOG_DIR/runner.log"
