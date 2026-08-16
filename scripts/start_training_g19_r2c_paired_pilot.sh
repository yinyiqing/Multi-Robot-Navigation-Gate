#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
TD3_DIR="$ROOT/TD3"
TRAIN_MANIFEST="$BASE/datasets/fixed_v1/views/g12_r3_mixed_v1/train.json.gz"
EVAL_MANIFEST="$BASE/datasets/fixed_v1/views/g12_full_scene_selection_v1/validation.json.gz"
SOURCE_ACTOR="$TD3_DIR/pytorch_models/TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best_actor.pth"
RUN_DIR="$BASE/19_R2C公平容量对照/local_data"
LOG_DIR="$ROOT/logs/active/g19-r2c-paired-pilot"
PID_FILE="$ROOT/.g19_r2c_paired_pilot.pid"
LOCK_FILE=/tmp/local_critic_multi_robot_training.lock
RESUME_ORIGINAL="${G19_RESUME_ORIGINAL:-0}"

verify_sha() {
  local path="$1" expected="$2"
  [[ -f "$path" ]] || { echo "Missing frozen input: $path" >&2; exit 1; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]] || {
    echo "SHA-256 mismatch: $path" >&2; exit 1
  }
}
verify_sha "$TRAIN_MANIFEST" c2ce37e51e8e98423d6ed6d295a7f5cf54d02e76c42f6459ce35003c899e0841
verify_sha "$EVAL_MANIFEST" 52435d6c5bdf9914e7212dd29cb4bfec074257f72d85f0d71741deee7c63b635
verify_sha "$SOURCE_ACTOR" fa28855049b67b3ee44c66d55d4f14441fc7c521e5429862c75b152f7d5cacc5

if [[ "${DRL_MULTI_DRY_RUN:-0}" == 1 ]]; then
  echo "G19-R2C dry run passed"
  echo "Order: original 60k -> stability gate -> wide 60k | resume_original=$RESUME_ORIGINAL"
  exit 0
fi
if [[ -f "$PID_FILE" ]]; then
  pid="$(tr -d '[:space:]' <"$PID_FILE")"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && {
    echo "G19-R2C already runs as PID $pid" >&2; exit 1
  }
  unlink "$PID_FILE"
fi
if [[ "$RESUME_ORIGINAL" == 1 ]]; then
  ORIGINAL_LATEST="$TD3_DIR/checkpoints/capacity_original_g19_r2c_n5_seed20260826_latest.pt"
  verify_sha "$ORIGINAL_LATEST" 3520feda0d552ca6a04bee8082c9fea47e99d3c0c1f5ec3b5a50976022efaa73
  (
    set +u
    source "$ROOT/env.python.sh" >/dev/null
    set -u
    python3 "$ROOT/scripts/audit_g19_r2c_resume.py" --checkpoint "$ORIGINAL_LATEST"
  )
  for suffix in latest best; do
    [[ ! -e "$TD3_DIR/checkpoints/capacity_wide_g19_r2c_n5_seed20260826_${suffix}.pt" ]] || {
      echo "G19-R2C wide artifact already exists: ${suffix}.pt" >&2; exit 1
    }
  done
else
  for model in capacity_original_g19_r2c_n5_seed20260826 capacity_wide_g19_r2c_n5_seed20260826; do
    for suffix in latest best; do
      [[ ! -e "$TD3_DIR/checkpoints/${model}_${suffix}.pt" ]] || {
        echo "G19-R2C artifact already exists: ${model}_${suffix}.pt" >&2; exit 1
      }
    done
  done
fi
[[ ! -e "$ROOT/logs/archive/training/g19_r2c_paired_pilot" ]] || {
  echo "G19-R2C archive already exists" >&2; exit 1
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

mkdir -p "$LOG_DIR" "$RUN_DIR"
/usr/bin/python3 "$ROOT/scripts/generate_multi_robot_launch.py" --num-agents 5 \
  --output "$LOG_DIR/runtime_g19_r2c.launch"
(
  set +u
  source "$ROOT/env.python.sh" >/dev/null
  set -u
  cd "$TD3_DIR"
  python3 "$ROOT/scripts/audit_g12_r2b_5a_recipe.py" \
    --source-actor "$SOURCE_ACTOR" --output "$RUN_DIR/initialization_audit.json"
)
G19_RESUME_ORIGINAL="$RESUME_ORIGINAL" \
  setsid bash "$ROOT/scripts/run_training_g19_r2c_paired_pilot_worker.sh" \
  >>"$LOG_DIR/runner.log" 2>&1 < /dev/null &
echo $! >"$PID_FILE"
echo "Started G19-R2C paired stability pilot"
echo "PID: $(cat "$PID_FILE")"
echo "GPU: 0 (free=${gpu_free_mib}MiB util=${gpu_util}%)"
echo "Order: original 60k -> gate -> wide 60k | resume_original=$RESUME_ORIGINAL"
echo "Log: $LOG_DIR/runner.log"
