#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
VIEW_ROOT="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views"
TRAIN_MANIFEST="$VIEW_ROOT/g12_r3_mixed_v1/train.json.gz"
EVAL_MANIFEST="$VIEW_ROOT/g12_full_scene_selection_v1/validation.json.gz"
LOAD_MODEL="capacity_wide_r2_s4_broad_n5_seed20260816_epoch_001"
MODEL_NAME="capacity_wide_r3_mixed_n5_seed20260818"
PID_FILE="$PROJECT_ROOT/.g12_r3_40k.pid"
LOG_DIR="$PROJECT_ROOT/logs/active/capacity-wide-g12-r3"
LAUNCHFILE="$LOG_DIR/runtime_g12_r3_40k.launch"
ROS_PORT=15451
GAZEBO_PORT=15551

declare -A EXPECTED_SHA256=(
  ["$TRAIN_MANIFEST"]="c2ce37e51e8e98423d6ed6d295a7f5cf54d02e76c42f6459ce35003c899e0841"
  ["$EVAL_MANIFEST"]="52435d6c5bdf9914e7212dd29cb4bfec074257f72d85f0d71741deee7c63b635"
  ["$TD3_DIR/pytorch_models/${LOAD_MODEL}_actor.pth"]="ace910553931873a275d66e3a964fd2b4716d30b6c68c8dcb3e7af96e56783ee"
)
for path in "${!EXPECTED_SHA256[@]}"; do
  [[ -f "$path" ]] || { echo "Required R3 input is missing: $path" >&2; exit 1; }
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "${EXPECTED_SHA256[$path]}" ]] || {
    echo "R3 input hash mismatch: $path" >&2
    exit 1
  }
done

if [[ "${DRL_MULTI_DRY_RUN:-0}" == 1 ]]; then
  echo "G12-R3 40k pilot dry run passed."
  echo "Warm start: $LOAD_MODEL (Actor only; fresh local Critic)"
  echo "Budget: 2 x 20000 = 40000 agent samples"
  echo "Schedule: standard/strong/dense/strong cycle"
  exit 0
fi

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "G12-R3 is already running with PID $old_pid" >&2
    exit 1
  fi
  unlink "$PID_FILE"
fi
for artifact in \
  "$TD3_DIR/checkpoints/${MODEL_NAME}_latest.pt" \
  "$TD3_DIR/pytorch_models/${MODEL_NAME}_actor.pth" \
  "$TD3_DIR/results/${MODEL_NAME}.npy"; do
  [[ ! -e "$artifact" ]] || { echo "Fresh R3 output already exists: $artifact" >&2; exit 1; }
done
if pgrep -af '^python3(\.8)? -u (train|test)_velodyne_td3_multi.py($| )' >/dev/null; then
  echo "Another multi-robot training or evaluation process is running" >&2
  exit 1
fi
if pgrep -af '(^|/)(gzserver|rosmaster)( |$)' >/dev/null; then
  echo "An existing Gazebo or ROS master is running; R3 will not start another" >&2
  exit 1
fi
if ! flock -n /tmp/local_critic_multi_robot_training.lock -c true; then
  echo "Another multi-robot process holds the training lock" >&2
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
  --num-agents 5 --output "$LAUNCHFILE"
timestamp="$(date +%Y%m%d_%H%M%S)"
train_log="$LOG_DIR/train_${MODEL_NAME}_${timestamp}.log"
runner_log="$LOG_DIR/train_${MODEL_NAME}_${timestamp}_runner.log"

setsid bash "$PROJECT_ROOT/scripts/run_training_g12_r3_40k_worker.sh" "$train_log" \
  >>"$runner_log" 2>&1 < /dev/null &
echo $! > "$PID_FILE"

echo "Started G12-R3 40k pilot."
echo "PID: $(cat "$PID_FILE")"
echo "GPU: 0 (free=${gpu_free_mib}MiB util=${gpu_util}%)"
echo "Budget: 40000 agent samples; eval=120 every 20k"
echo "Training log: $train_log"
echo "Runner log: $runner_log"
