#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
MANIFEST="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/g12_r2_curriculum_v1/n5/validation.json.gz"
N5_ACTOR="$TD3_DIR/pytorch_models/current_generalist_n5_original_broad_s20260810_best_actor.pth"
REFERENCE_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/12_参数匹配单Actor容量对照/local_data/r2_n5_admission/results"
BASE_5A="$REFERENCE_DIR/g12_r2_n5_admission_5a_s20260817.npy"
BASE_R2="$REFERENCE_DIR/g12_r2_n5_admission_r2_10k_s20260817.npy"
PID_FILE="$PROJECT_ROOT/.current_generalist_n5_admission.pid"
LOG_DIR="$PROJECT_ROOT/logs/active/current-generalist-r2style/n5-admission"
RUNNER_LOG="$LOG_DIR/runner.log"
ROS_PORT=15451
GAZEBO_PORT=15551

declare -A EXPECTED_SHA256=(
  ["$MANIFEST"]="e33dbfad3d166fa4500b5997902a94c49108c77c646c7a39c26480b5054daef7"
  ["$N5_ACTOR"]="53964e12c2d6c5f0855530f22bdd721170b911640883c7616b14dc21aa12cfeb"
)
for path in "${!EXPECTED_SHA256[@]}"; do
  [[ -f "$path" ]] || { echo "Admission input is missing: $path" >&2; exit 1; }
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "${EXPECTED_SHA256[$path]}" ]] || {
    echo "Admission input hash mismatch: $path" >&2
    echo "expected=${EXPECTED_SHA256[$path]} actual=$actual" >&2
    exit 1
  }
done
for path in "$BASE_5A" "$BASE_R2"; do
  [[ -f "$path" ]] || {
    echo "Audited reference result is missing: $path" >&2
    exit 1
  }
done

if [[ "${DRL_MULTI_DRY_RUN:-0}" == 1 ]]; then
  echo "Current-generalist N5 paired admission dry run passed."
  echo "Candidate: current_generalist_n5_original_broad_s20260810_best"
  echo "References: audited 5A and R2-10k, seed=20260817"
  echo "Episodes: 120"
  exit 0
fi

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Current-generalist N5 admission is already running with PID $old_pid" >&2
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
if (( gpu_free_mib < 4096 || gpu_util > 35 )); then
  echo "GPU 0 is not available enough: free=${gpu_free_mib}MiB util=${gpu_util}%" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
/usr/bin/python3 "$PROJECT_ROOT/scripts/generate_multi_robot_launch.py" \
  --num-agents 5 --output "$LOG_DIR/runtime_current_generalist_n5_admission.launch"
setsid bash "$PROJECT_ROOT/scripts/run_current_generalist_n5_admission_worker.sh" \
  >>"$RUNNER_LOG" 2>&1 < /dev/null &
echo $! > "$PID_FILE"

echo "Started current-generalist N5 paired admission."
echo "PID: $(cat "$PID_FILE")"
echo "Candidate: N5-20k best; 120 episodes"
echo "GPU: 0 (free=${gpu_free_mib}MiB util=${gpu_util}%)"
echo "Runner log: $RUNNER_LOG"
