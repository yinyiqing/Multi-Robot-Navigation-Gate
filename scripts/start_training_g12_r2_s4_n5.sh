#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
VIEW_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/g12_r2_curriculum_v1/n5"
TRAIN_MANIFEST="$VIEW_DIR/train.json.gz"
EVAL_MANIFEST="$VIEW_DIR/validation.json.gz"
MODEL_NAME="capacity_wide_r2_s4_broad_n5_seed20260816"
LOAD_MODEL="capacity_wide_r2_s3_broad_n3_seed20260815_best"
PID_FILE="$PROJECT_ROOT/.g12_r2_s4_n5.pid"
LOG_DIR="$PROJECT_ROOT/logs/active/capacity-wide-g12-r2/s4-n5"
LAUNCHFILE="$LOG_DIR/runtime_g12_r2_s4_n5.launch"
ROS_PORT=15051
GAZEBO_PORT=15151

declare -A EXPECTED_SHA256=(
  ["$TRAIN_MANIFEST"]="82f990dab54331ef55d3818fbe39b31fe00480dd99696987a5b85c5e2581ac1e"
  ["$EVAL_MANIFEST"]="e33dbfad3d166fa4500b5997902a94c49108c77c646c7a39c26480b5054daef7"
  ["$TD3_DIR/pytorch_models/${LOAD_MODEL}_actor.pth"]="0ad69f89378b88812c1ce2306a07c75fbd4d80a9616b1db3a18e6d36c9037f04"
  ["$TD3_DIR/pytorch_models/${LOAD_MODEL}_critic.pth"]="55a20491f6f498960d77284e44409c99d7d710bb5a39fb18a212a3d047650d67"
)
for path in "${!EXPECTED_SHA256[@]}"; do
  [[ -f "$path" ]] || { echo "Required S4 input is missing: $path" >&2; exit 1; }
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "${EXPECTED_SHA256[$path]}" ]] || {
    echo "S4 input hash mismatch: $path" >&2
    exit 1
  }
done

if [[ "${DRL_MULTI_DRY_RUN:-0}" == 1 ]]; then
  echo "G12-R2-S4 n5 pilot dry run passed."
  echo "Model: $MODEL_NAME"
  echo "Warm start: $LOAD_MODEL (Actor and Critic)"
  echo "Agents: 5"
  echo "Budget: 2 x 10000 = 20000 agent samples"
  echo "Validation: 120 fixed n5 episodes every 10k"
  exit 0
fi

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "G12-R2-S4 n5 is already running with PID $old_pid" >&2
    exit 1
  fi
  unlink "$PID_FILE"
fi
for artifact in \
  "$TD3_DIR/checkpoints/${MODEL_NAME}_latest.pt" \
  "$TD3_DIR/pytorch_models/${MODEL_NAME}_actor.pth" \
  "$TD3_DIR/results/${MODEL_NAME}.npy"; do
  [[ ! -e "$artifact" ]] || { echo "Fresh S4 output already exists: $artifact" >&2; exit 1; }
done
if pgrep -af '^python3(\.8)? -u (train|test)_velodyne_td3_multi.py($| )' >/dev/null; then
  echo "Another multi-robot training or evaluation process is running" >&2
  exit 1
fi
if pgrep -af '(^|/)(gzserver|rosmaster)( |$)' >/dev/null; then
  echo "An existing Gazebo or ROS master is running; S4 will not start another" >&2
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

setsid bash "$PROJECT_ROOT/scripts/run_training_g12_r2_s4_n5_worker.sh" "$train_log" \
  >>"$runner_log" 2>&1 < /dev/null &
echo $! > "$PID_FILE"

echo "Started G12-R2-S4 five-robot pilot."
echo "PID: $(cat "$PID_FILE")"
echo "GPU: 0 (free=${gpu_free_mib}MiB util=${gpu_util}%)"
echo "Budget: 20000 agent samples; eval=120 every 10k"
echo "Training log: $train_log"
echo "Runner log: $runner_log"
