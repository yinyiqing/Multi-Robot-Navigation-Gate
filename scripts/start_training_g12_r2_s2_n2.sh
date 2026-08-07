#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
VIEW_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/g12_r2_curriculum_v1/n2"
TRAIN_MANIFEST="$VIEW_DIR/train.json.gz"
EVAL_MANIFEST="$VIEW_DIR/validation.json.gz"
MODEL_NAME="capacity_wide_r2_s2_broad_n2_seed20260814"
LOAD_MODEL="capacity_wide_r2_s0_broad_n1_seed20260811_best"
PID_FILE="$PROJECT_ROOT/.g12_r2_s2_n2.pid"
LOG_DIR="$PROJECT_ROOT/logs/active/capacity-wide-g12-r2/s2-n2"
LAUNCHFILE="$LOG_DIR/runtime_g12_r2_s2_n2.launch"
ROS_PORT=14651
GAZEBO_PORT=14751

declare -A EXPECTED_SHA256=(
  ["$TRAIN_MANIFEST"]="5fbd2df5241076041ea714b59286604915ebf1b13848482f7c34fd10cdc9087b"
  ["$EVAL_MANIFEST"]="955132263cac9496a56eb8bb6f5132ca5ae41e930c926a7a9a13e8797bb903c9"
  ["$TD3_DIR/pytorch_models/${LOAD_MODEL}_actor.pth"]="7cb61925a4188e638859f88d38288e0431e5f05be489fa6107a77c7efaed3822"
  ["$TD3_DIR/pytorch_models/${LOAD_MODEL}_critic.pth"]="1d5e9bbcc7062886548cf2691ce446993bb5a04be11d5a517cbcb9fa610ad752"
)
for path in "${!EXPECTED_SHA256[@]}"; do
  [[ -f "$path" ]] || { echo "Required S2 input is missing: $path" >&2; exit 1; }
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "${EXPECTED_SHA256[$path]}" ]] || {
    echo "S2 input hash mismatch: $path" >&2
    exit 1
  }
done

if [[ "${DRL_MULTI_DRY_RUN:-0}" == 1 ]]; then
  echo "G12-R2-S2 n2 pilot dry run passed."
  echo "Model: $MODEL_NAME"
  echo "Warm start: $LOAD_MODEL (Actor and Critic)"
  echo "Agents: 2"
  echo "Budget: 2 x 10000 = 20000 agent samples"
  echo "Validation: 120 fixed n2 episodes every 10k"
  exit 0
fi

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "G12-R2-S2 n2 is already running with PID $old_pid" >&2
    exit 1
  fi
  unlink "$PID_FILE"
fi
for artifact in \
  "$TD3_DIR/checkpoints/${MODEL_NAME}_latest.pt" \
  "$TD3_DIR/pytorch_models/${MODEL_NAME}_actor.pth" \
  "$TD3_DIR/results/${MODEL_NAME}.npy"; do
  [[ ! -e "$artifact" ]] || { echo "Fresh S2 output already exists: $artifact" >&2; exit 1; }
done
if pgrep -af '^python3(\.8)? -u (train|test)_velodyne_td3_multi.py($| )' >/dev/null; then
  echo "Another multi-robot training or evaluation process is running" >&2
  exit 1
fi
if pgrep -af '(^|/)(gzserver|rosmaster)( |$)' >/dev/null; then
  echo "An existing Gazebo or ROS master is running; S2 will not start another" >&2
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
  --num-agents 2 --output "$LAUNCHFILE"
timestamp="$(date +%Y%m%d_%H%M%S)"
train_log="$LOG_DIR/train_${MODEL_NAME}_${timestamp}.log"
runner_log="$LOG_DIR/train_${MODEL_NAME}_${timestamp}_runner.log"

setsid bash "$PROJECT_ROOT/scripts/run_training_g12_r2_s2_n2_worker.sh" "$train_log" \
  >>"$runner_log" 2>&1 < /dev/null &
echo $! > "$PID_FILE"

echo "Started G12-R2-S2 two-robot pilot."
echo "PID: $(cat "$PID_FILE")"
echo "GPU: 0 (free=${gpu_free_mib}MiB util=${gpu_util}%)"
echo "Budget: 20000 agent samples; eval=120 every 10k"
echo "Training log: $train_log"
echo "Runner log: $runner_log"
