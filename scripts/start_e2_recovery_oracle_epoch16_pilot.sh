#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
MANIFEST="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/g12_r2_curriculum_v1/n5/validation.json.gz"
E2_ACTOR="$TD3_DIR/pytorch_models/current_generalist_n5_efficiency_e2_s20260810_best_actor.pth"
INTERACTION_ACTOR="$TD3_DIR/pytorch_models/interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016_actor.pth"
PID_FILE="$PROJECT_ROOT/.e2_recovery_oracle_epoch16_pilot.pid"
LOG_DIR="$PROJECT_ROOT/logs/active/current-generalist-r2style/e2-recovery-oracle-epoch16-pilot"
RUNNER_LOG="$LOG_DIR/runner.log"
ROS_PORT=15457
GAZEBO_PORT=15557

declare -A EXPECTED_SHA256=(
  ["$MANIFEST"]="e33dbfad3d166fa4500b5997902a94c49108c77c646c7a39c26480b5054daef7"
  ["$E2_ACTOR"]="dbfd69a27aafae5cbb4686018bc074d3684de45ff397f1dffe77f5265138c7d0"
  ["$INTERACTION_ACTOR"]="6ec1942fcd497ab1cc2a85a5aaec8f524395dc21ff21a442dca243a52e917c0b"
)
for path in "${!EXPECTED_SHA256[@]}"; do
  [[ -f "$path" ]] || { echo "Recovery-oracle input is missing: $path" >&2; exit 1; }
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "${EXPECTED_SHA256[$path]}" ]] || {
    echo "Recovery-oracle input hash mismatch: $path" >&2
    echo "expected=${EXPECTED_SHA256[$path]} actual=$actual" >&2
    exit 1
  }
done

if [[ "${DRL_MULTI_DRY_RUN:-0}" == 1 ]]; then
  echo "E2 recovery-oracle epoch16 pilot dry run passed."
  echo "Standard Actor: current_generalist_n5_efficiency_e2_s20260810_best"
  echo "Interaction Actor: interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016"
  echo "Episodes: 120"
  echo "Logs: $LOG_DIR"
  exit 0
fi

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "E2 recovery-oracle pilot is already running with PID $old_pid" >&2
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
  --num-agents 5 --output "$LOG_DIR/runtime_e2_recovery_oracle_epoch16_pilot.launch"
setsid bash "$PROJECT_ROOT/scripts/run_e2_recovery_oracle_epoch16_pilot_worker.sh" \
  >>"$RUNNER_LOG" 2>&1 < /dev/null &
echo $! > "$PID_FILE"

echo "Started E2 recovery-oracle epoch16 pilot."
echo "PID: $(cat "$PID_FILE")"
echo "Oracle: E2 normally, epoch16 only under near-robot + stagnation recovery rule"
echo "GPU: 0 (free=${gpu_free_mib}MiB util=${gpu_util}%)"
echo "Runner log: $RUNNER_LOG"
