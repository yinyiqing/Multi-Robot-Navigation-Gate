#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
CASE_FILE="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/g12_r2_curriculum_v1/s1_repair_train_cases.json"
EVAL_MANIFEST="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/g12_r2_curriculum_v1/n1/validation.json.gz"
MODEL_NAME="capacity_wide_r2_s1_repair_n1_seed20260813"
LOAD_MODEL="capacity_wide_r2_s0_broad_n1_seed20260811_best"
PID_FILE="$PROJECT_ROOT/.g12_r2_s1_repair.pid"
LOCK_FILE="/tmp/local_critic_multi_robot_training.lock"
LOG_DIR="$PROJECT_ROOT/logs/active/capacity-wide-g12-r2/s1-repair"
LAUNCHFILE="$LOG_DIR/runtime_g12_r2_s1_repair.launch"
ROS_PORT=14631
GAZEBO_PORT=14731

declare -A EXPECTED_SHA256=(
  ["$CASE_FILE"]="df5267aac5b671befe1df2c64d82503bda9acd4770b02e774e02717d32a9ace5"
  ["$EVAL_MANIFEST"]="9ab4c5913f683d01e3ab186ea591d373abe1e835180f4a0bfeb469990269b125"
  ["$TD3_DIR/pytorch_models/${LOAD_MODEL}_actor.pth"]="7cb61925a4188e638859f88d38288e0431e5f05be489fa6107a77c7efaed3822"
  ["$TD3_DIR/pytorch_models/${LOAD_MODEL}_critic.pth"]="1d5e9bbcc7062886548cf2691ce446993bb5a04be11d5a517cbcb9fa610ad752"
)

for path in "${!EXPECTED_SHA256[@]}"; do
  [[ -f "$path" ]] || { echo "Required S1 input is missing: $path" >&2; exit 1; }
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "${EXPECTED_SHA256[$path]}" ]] || {
    echo "S1 input hash mismatch: $path" >&2
    echo "expected=${EXPECTED_SHA256[$path]} actual=$actual" >&2
    exit 1
  }
done

/usr/bin/python3 - "$CASE_FILE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    cases = json.load(handle)["cases"]
if len(cases) != 8 or len({item["name"] for item in cases}) != 8:
    raise SystemExit("S1 repair file must contain eight unique cases")
if abs(sum(float(item["weight"]) for item in cases) - 12.0) > 1e-9:
    raise SystemExit("S1 repair weights do not match the frozen protocol")
PY

if [[ "${DRL_MULTI_DRY_RUN:-0}" == 1 ]]; then
  cat <<EOF
G12-R2-S1 repair pilot dry run passed.
Model: $MODEL_NAME
Warm start: $LOAD_MODEL (Actor and Critic)
Actor: 24->1137->855->2
Repair cases: 8 unique geometries, weighted random sampling
Budget: 20000 agent samples
Broad validation: 120 fixed n1 episodes
EOF
  exit 0
fi

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "G12-R2-S1 repair pilot is already running with PID $old_pid" >&2
    exit 1
  fi
  unlink "$PID_FILE"
fi
for artifact in \
  "$TD3_DIR/checkpoints/${MODEL_NAME}_latest.pt" \
  "$TD3_DIR/pytorch_models/${MODEL_NAME}_actor.pth" \
  "$TD3_DIR/results/${MODEL_NAME}.npy"; do
  [[ ! -e "$artifact" ]] || { echo "Fresh S1 output already exists: $artifact" >&2; exit 1; }
done
if pgrep -af '^python3(\.8)? -u (train|test)_velodyne_td3_multi.py($| )' >/dev/null; then
  echo "Another multi-robot training or evaluation process is running" >&2
  exit 1
fi
if pgrep -af '(^|/)(gzserver|rosmaster)( |$)' >/dev/null; then
  echo "An existing Gazebo or ROS master is running; S1 will not start another" >&2
  exit 1
fi
if ! flock -n "$LOCK_FILE" -c true; then
  echo "Another multi-robot process holds $LOCK_FILE" >&2
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
  --num-agents 1 --output "$LAUNCHFILE"
timestamp="$(date +%Y%m%d_%H%M%S)"
train_log="$LOG_DIR/train_${MODEL_NAME}_${timestamp}.log"
runner_log="$LOG_DIR/train_${MODEL_NAME}_${timestamp}_runner.log"

setsid bash "$PROJECT_ROOT/scripts/run_training_g12_r2_s1_repair_worker.sh" "$train_log" \
  >>"$runner_log" 2>&1 < /dev/null &
echo $! > "$PID_FILE"

echo "Started G12-R2-S1 repair pilot."
echo "PID: $(cat "$PID_FILE")"
echo "GPU: 0 (free=${gpu_free_mib}MiB util=${gpu_util}%)"
echo "Budget: 20000 agent samples; broad eval=120"
echo "Training log: $train_log"
echo "Runner log: $runner_log"
