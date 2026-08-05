#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究/G11_C_50场闭环pilot"
PID_FILE="$PROJECT_ROOT/.g11_c_pilot.pid"
ACTIVE_LOG_DIR="$PROJECT_ROOT/local/logs/gate-g11-c-pilot"
RUNNER_LOG="$ACTIVE_LOG_DIR/pilot_runner.log"
ROS_PORT=14623
GAZEBO_PORT=14723

declare -A EXPECTED_SHA=(
  ["$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/g11_c_pilot_v1/validation.json.gz"]="1bf044cb5ff9d7d80c14d860d1108481af1d422cf403b26869f8b963012f0e91"
  ["$PROJECT_ROOT/TD3/pytorch_models/TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best_actor.pth"]="fa28855049b67b3ee44c66d55d4f14441fc7c521e5429862c75b152f7d5cacc5"
  ["$PROJECT_ROOT/TD3/pytorch_models/interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016_actor.pth"]="6ec1942fcd497ab1cc2a85a5aaec8f524395dc21ff21a442dca243a52e917c0b"
  ["$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"]="0b914c0d090bbaba0a2be63c0d75d88580bf4d71778a7f1749c63989e3dbbd56"
  ["$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究/G11_A1_当前协议时序pilot/local_data/training/seed20260804/any/T1/best.pt"]="d9b05d9f86e5bad4d2071c041187b618ebca6f1a3cc1f9c46e8b14b1a451537a"
  ["$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究/G11_B_student_rollout_v1/local_data/training/seed20260804/any/T1/best.pt"]="fc59b4f783f7c5461ebb0239fab4b34896ad910ee78e7223e88d29ce9c3f5a52"
)
for path in "${!EXPECTED_SHA[@]}"; do
  [[ -f "$path" ]] || { echo "Required input is missing: $path" >&2; exit 1; }
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "${EXPECTED_SHA[$path]}" ]] || {
    echo "Frozen input hash mismatch: $path" >&2
    exit 1
  }
done

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "G11-C pilot is already running with PID $old_pid" >&2
    exit 1
  fi
  unlink "$PID_FILE"
fi
for port in "$ROS_PORT" "$GAZEBO_PORT"; do
  if ss -ltnH | awk '{print $4}' | rg -q ":${port}$"; then
    echo "Port $port is already in use" >&2
    exit 1
  fi
done

mkdir -p "$RUN_DIR/local_data" "$ACTIVE_LOG_DIR"
setsid bash "$PROJECT_ROOT/scripts/run_g11_c_pilot_worker.sh" \
  >>"$RUNNER_LOG" 2>&1 < /dev/null &
echo $! > "$PID_FILE"

echo "Started G11-C fixed 50-scene paired pilot."
echo "PID: $(cat "$PID_FILE")"
echo "Runs: 5A/A1/B2 x 2 repeats, serial, CPU only"
echo "Runner log: $RUNNER_LOG"
