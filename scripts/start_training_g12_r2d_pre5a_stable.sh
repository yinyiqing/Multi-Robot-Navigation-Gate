#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$ROOT/TD3"
SOURCE="$TD3_DIR/checkpoints/capacity_wide_r2b_5a_recipe_n5_seed20260823_best.pt"
MODEL_NAME="capacity_wide_r2d_pre5a_stable_n5_seed20260823"
OUTPUT="$TD3_DIR/checkpoints/${MODEL_NAME}_latest.pt"
EVAL_MANIFEST="$ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/g12_r2_curriculum_v1/n5/validation.json.gz"
LOG_DIR="$ROOT/logs/active/capacity-wide-g12-r2d-pre5a-stable"
PID_FILE="$ROOT/.g12_r2d_pre5a_stable.pid"
LOCK_FILE=/tmp/local_critic_multi_robot_training.lock

verify_sha() {
  local path="$1" expected="$2"
  [[ -f "$path" ]] || { echo "Missing frozen input: $path" >&2; exit 1; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]] || {
    echo "SHA-256 mismatch: $path" >&2; exit 1
  }
}
verify_sha "$SOURCE" 6e4f47e2665d5040f3962e0283000ecdf4c9a6fb03b00ce3b2c45eda95b5ec60
verify_sha "$EVAL_MANIFEST" e33dbfad3d166fa4500b5997902a94c49108c77c646c7a39c26480b5054daef7

if [[ -f "$PID_FILE" ]]; then
  pid="$(tr -d '[:space:]' <"$PID_FILE")"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && {
    echo "R2D already runs as PID $pid" >&2; exit 1
  }
  unlink "$PID_FILE"
fi
[[ ! -e "$OUTPUT" ]] || { echo "R2D output already exists: $OUTPUT" >&2; exit 1; }
[[ ! -e "$ROOT/logs/archive/training/capacity_wide_g12_r2d_pre5a_stable" ]] || {
  echo "R2D archive already exists" >&2; exit 1
}
if pgrep -af '^python3(\.8)? -u (train|test)_velodyne_td3_multi.py($| )' >/dev/null; then
  echo "Another multi-robot run is active" >&2; exit 1
fi
if pgrep -af '(^|/)(gzserver|rosmaster)( |$)' >/dev/null; then
  echo "An existing Gazebo or ROS master is active" >&2; exit 1
fi
flock -n "$LOCK_FILE" -c true || { echo "Multi-robot lock is busy" >&2; exit 1; }
gpu_free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')"
gpu_util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 | tr -d ' ')"
[[ "$gpu_free_mib" -ge 8192 && "$gpu_util" -le 20 ]] || {
  echo "GPU 0 is too busy: free=${gpu_free_mib}MiB util=${gpu_util}%" >&2; exit 1
}

mkdir -p "$LOG_DIR"
/usr/bin/python3 "$ROOT/scripts/generate_multi_robot_launch.py" --num-agents 5 \
  --output "$LOG_DIR/runtime_g12_r2d_pre5a_stable.launch"
set +u
source "$ROOT/env.python.sh" >/dev/null
set -u
python3 "$ROOT/scripts/fork_r2b_stable_checkpoint.py" --source "$SOURCE" --output "$OUTPUT"
G12_R2D=1 setsid bash "$ROOT/scripts/run_training_g12_r2d_pre5a_stable_worker.sh" \
  >>"$LOG_DIR/runner.log" 2>&1 < /dev/null &
echo $! >"$PID_FILE"
echo "Started R2D pre-5A stable continuation"
echo "PID: $(cat "$PID_FILE")"
echo "GPU: 0 (free=${gpu_free_mib}MiB util=${gpu_util}%)"
echo "Log: $LOG_DIR/train_${MODEL_NAME}.log"
