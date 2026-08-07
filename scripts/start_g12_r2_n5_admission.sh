#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
MANIFEST="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/g12_r2_curriculum_v1/n5/validation.json.gz"
BASE_ACTOR="$TD3_DIR/pytorch_models/TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best_actor.pth"
R2_ACTOR="$TD3_DIR/pytorch_models/capacity_wide_r2_s4_broad_n5_seed20260816_best_actor.pth"
PID_FILE="$PROJECT_ROOT/.g12_r2_n5_admission.pid"
LOG_DIR="$PROJECT_ROOT/logs/active/capacity-wide-g12-r2/n5-admission"
RUNNER_LOG="$LOG_DIR/runner.log"
ROS_PORT=15251
GAZEBO_PORT=15351

declare -A EXPECTED_SHA256=(
  ["$MANIFEST"]="e33dbfad3d166fa4500b5997902a94c49108c77c646c7a39c26480b5054daef7"
  ["$BASE_ACTOR"]="fa28855049b67b3ee44c66d55d4f14441fc7c521e5429862c75b152f7d5cacc5"
  ["$R2_ACTOR"]="67290450484c1fedd493fb029804b914438c5fb46cdb189ba8c642c3d98b2715"
)
for path in "${!EXPECTED_SHA256[@]}"; do
  [[ -f "$path" ]] || { echo "Admission input is missing: $path" >&2; exit 1; }
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "${EXPECTED_SHA256[$path]}" ]] || {
    echo "Admission input hash mismatch: $path" >&2
    exit 1
  }
done

if [[ "${DRL_MULTI_DRY_RUN:-0}" == 1 ]]; then
  echo "G12-R2 N5 paired admission dry run passed."
  echo "Policies: 5A and R2-S4 20k best"
  echo "Episodes: 120 each; seed=20260817"
  exit 0
fi

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "G12-R2 N5 admission is already running with PID $old_pid" >&2
    exit 1
  fi
  unlink "$PID_FILE"
fi
if pgrep -af '^python3(\.8)? -u (train|test)_velodyne_td3_multi.py($| )' >/dev/null; then
  echo "Another multi-robot training or evaluation process is running" >&2
  exit 1
fi
if pgrep -af '(^|/)(gzserver|rosmaster)( |$)' >/dev/null; then
  echo "An existing Gazebo or ROS master is running" >&2
  exit 1
fi
if ! flock -n /tmp/local_critic_multi_robot_training.lock -c true; then
  echo "Another multi-robot process holds the evaluation lock" >&2
  exit 1
fi
for port in "$ROS_PORT" "$GAZEBO_PORT"; do
  if ss -ltnH | awk '{print $4}' | grep -Eq ":${port}$"; then
    echo "Port $port is already in use" >&2
    exit 1
  fi
done

gpu_free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d '[:space:]')"
gpu_util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 | tr -d '[:space:]')"
if (( gpu_free_mib < 8192 || gpu_util > 20 )); then
  echo "GPU 0 is not available enough: free=${gpu_free_mib}MiB util=${gpu_util}%" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
/usr/bin/python3 "$PROJECT_ROOT/scripts/generate_multi_robot_launch.py" \
  --num-agents 5 --output "$LOG_DIR/runtime_g12_r2_n5_admission.launch"
setsid bash "$PROJECT_ROOT/scripts/run_g12_r2_n5_admission_worker.sh" \
  >>"$RUNNER_LOG" 2>&1 < /dev/null &
echo $! > "$PID_FILE"

echo "Started G12-R2 N5 paired admission."
echo "PID: $(cat "$PID_FILE")"
echo "Policies: 5A -> R2-S4; 120 episodes each"
echo "GPU: 0 (free=${gpu_free_mib}MiB util=${gpu_util}%)"
echo "Runner log: $RUNNER_LOG"

