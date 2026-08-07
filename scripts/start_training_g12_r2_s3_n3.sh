#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
VIEW_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/g12_r2_curriculum_v1/n3"
TRAIN_MANIFEST="$VIEW_DIR/train.json.gz"
EVAL_MANIFEST="$VIEW_DIR/validation.json.gz"
MODEL_NAME="capacity_wide_r2_s3_broad_n3_seed20260815"
LOAD_MODEL="capacity_wide_r2_s2_broad_n2_seed20260814_best"
PID_FILE="$PROJECT_ROOT/.g12_r2_s3_n3.pid"
LOG_DIR="$PROJECT_ROOT/logs/active/capacity-wide-g12-r2/s3-n3"
LAUNCHFILE="$LOG_DIR/runtime_g12_r2_s3_n3.launch"
ROS_PORT=14851
GAZEBO_PORT=14951

declare -A EXPECTED_SHA256=(
  ["$TRAIN_MANIFEST"]="b6ff22964a8b1795a783f8af9360c123fae44b4b44a86de63e76a57b4a0b4422"
  ["$EVAL_MANIFEST"]="f4b7d46fc488eb588007aa7ba72791545e750e691399da82c65d5cdf9f5938cc"
  ["$TD3_DIR/pytorch_models/${LOAD_MODEL}_actor.pth"]="220698f1e4a918deb88d0b47f8c4f28b2330194401b4b82c80afe92d8f63f465"
  ["$TD3_DIR/pytorch_models/${LOAD_MODEL}_critic.pth"]="acbecd846cbae2456e2a493ec545eeaf2718a11aa2cc6fe58c2a9d3af1fbe7ca"
)
for path in "${!EXPECTED_SHA256[@]}"; do
  [[ -f "$path" ]] || { echo "Required S3 input is missing: $path" >&2; exit 1; }
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "${EXPECTED_SHA256[$path]}" ]] || {
    echo "S3 input hash mismatch: $path" >&2
    exit 1
  }
done

if [[ "${DRL_MULTI_DRY_RUN:-0}" == 1 ]]; then
  echo "G12-R2-S3 n3 pilot dry run passed."
  echo "Model: $MODEL_NAME"
  echo "Warm start: $LOAD_MODEL (Actor and Critic)"
  echo "Agents: 3"
  echo "Budget: 2 x 10000 = 20000 agent samples"
  echo "Validation: 120 fixed n3 episodes every 10k"
  exit 0
fi

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "G12-R2-S3 n3 is already running with PID $old_pid" >&2
    exit 1
  fi
  unlink "$PID_FILE"
fi
for artifact in \
  "$TD3_DIR/checkpoints/${MODEL_NAME}_latest.pt" \
  "$TD3_DIR/pytorch_models/${MODEL_NAME}_actor.pth" \
  "$TD3_DIR/results/${MODEL_NAME}.npy"; do
  [[ ! -e "$artifact" ]] || { echo "Fresh S3 output already exists: $artifact" >&2; exit 1; }
done
if pgrep -af '^python3(\.8)? -u (train|test)_velodyne_td3_multi.py($| )' >/dev/null; then
  echo "Another multi-robot training or evaluation process is running" >&2
  exit 1
fi
if pgrep -af '(^|/)(gzserver|rosmaster)( |$)' >/dev/null; then
  echo "An existing Gazebo or ROS master is running; S3 will not start another" >&2
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
  --num-agents 3 --output "$LAUNCHFILE"
timestamp="$(date +%Y%m%d_%H%M%S)"
train_log="$LOG_DIR/train_${MODEL_NAME}_${timestamp}.log"
runner_log="$LOG_DIR/train_${MODEL_NAME}_${timestamp}_runner.log"

setsid bash "$PROJECT_ROOT/scripts/run_training_g12_r2_s3_n3_worker.sh" "$train_log" \
  >>"$runner_log" 2>&1 < /dev/null &
echo $! > "$PID_FILE"

echo "Started G12-R2-S3 three-robot pilot."
echo "PID: $(cat "$PID_FILE")"
echo "GPU: 0 (free=${gpu_free_mib}MiB util=${gpu_util}%)"
echo "Budget: 20000 agent samples; eval=120 every 10k"
echo "Training log: $train_log"
echo "Runner log: $runner_log"

