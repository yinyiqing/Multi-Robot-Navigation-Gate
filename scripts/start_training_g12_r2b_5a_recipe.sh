#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$ROOT/TD3"
MODEL_NAME="capacity_wide_r2b_5a_recipe_n5_seed20260823"
SOURCE_ACTOR="$TD3_DIR/pytorch_models/TD3_velodyne_multi_v4_curriculum_stage2_to_3d2_geo_critic_from_3a_guarded_best_actor.pth"
EVAL_MANIFEST="$ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/g12_r2_curriculum_v1/n5/validation.json.gz"
RUN_DIR="$ROOT/experiments/03_保留专门化/02_论文主线/12_参数匹配单Actor容量对照/local_data/r2b_5a_recipe"
LOG_DIR="$ROOT/logs/active/capacity-wide-g12-r2b-5a-recipe"
LAUNCHFILE="$LOG_DIR/runtime_g12_r2b_5a_recipe_n5.launch"
PID_FILE="$ROOT/.g12_r2b_5a_recipe.pid"
LOCK_FILE=/tmp/local_critic_multi_robot_training.lock

declare -A EXPECTED=(
  ["$SOURCE_ACTOR"]="9be0658c1f33505103f2a3e92714de3fd3759bf5d7eecec878657f43987333b5"
  ["$EVAL_MANIFEST"]="e33dbfad3d166fa4500b5997902a94c49108c77c646c7a39c26480b5054daef7"
)
for path in "${!EXPECTED[@]}"; do
  [[ -f "$path" ]] || { echo "Missing frozen R2B input: $path" >&2; exit 1; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "${EXPECTED[$path]}" ]] || {
    echo "Frozen R2B input hash mismatch: $path" >&2; exit 1
  }
done

if [[ "${DRL_MULTI_DRY_RUN:-0}" == 1 ]]; then
  echo "G12-R2B dry run passed"
  echo "Model: $MODEL_NAME"
  echo "Actor: 3D2 function-preserving expansion to 24->1137->855->2"
  echo "Training: five-robot procedural standard, 30k samples"
  echo "Validation: $EVAL_MANIFEST (120 episodes per 10k)"
  exit 0
fi

if [[ -f "$PID_FILE" ]]; then
  pid="$(tr -d '[:space:]' <"$PID_FILE")"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && {
    echo "G12-R2B already runs as PID $pid" >&2; exit 1
  }
  unlink "$PID_FILE"
fi
for suffix in latest best; do
  [[ ! -e "$TD3_DIR/checkpoints/${MODEL_NAME}_${suffix}.pt" ]] || {
    echo "R2B artifact already exists: ${MODEL_NAME}_${suffix}.pt" >&2; exit 1
  }
done
if pgrep -af '^python3(\.8)? -u (train|test)_velodyne_td3_multi.py($| )' >/dev/null; then
  echo "Another multi-robot run is active" >&2; exit 1
fi
if pgrep -af '(^|/)(gzserver|rosmaster)( |$)' >/dev/null; then
  echo "An existing Gazebo or ROS master is active" >&2; exit 1
fi
if ! flock -n "$LOCK_FILE" -c true; then
  echo "Multi-robot training lock is busy" >&2; exit 1
fi
for port in 17023 17123; do
  ss -ltnH | awk '{print $4}' | grep -Eq ":${port}$" && { echo "Port $port in use" >&2; exit 1; }
done
gpu_free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')"
gpu_util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 | tr -d ' ')"
[[ "$gpu_free_mib" -ge 8192 && "$gpu_util" -le 20 ]] || {
  echo "GPU 0 is too busy: free=${gpu_free_mib}MiB util=${gpu_util}%" >&2; exit 1
}

mkdir -p "$LOG_DIR" "$RUN_DIR"
/usr/bin/python3 "$ROOT/scripts/generate_multi_robot_launch.py" --num-agents 5 --output "$LAUNCHFILE"
(
  set +u
  source "$ROOT/env.python.sh" >/dev/null
  set -u
  cd "$TD3_DIR"
  python3 "$ROOT/scripts/audit_g12_r2b_5a_recipe.py" \
    --source-actor "$SOURCE_ACTOR" --output "$RUN_DIR/initialization_audit.json"
)
timestamp="$(date +%Y%m%d_%H%M%S)"
train_log="$LOG_DIR/train_${MODEL_NAME}_${timestamp}.log"
runner_log="$LOG_DIR/train_${MODEL_NAME}_${timestamp}_runner.log"
setsid bash "$ROOT/scripts/run_training_g12_r2b_5a_recipe_worker.sh" "$train_log" >>"$runner_log" 2>&1 < /dev/null &
echo $! >"$PID_FILE"
echo "Started G12-R2B 5A-recipe wide Actor"
echo "PID: $(cat "$PID_FILE")"
echo "GPU: 0 (free=${gpu_free_mib}MiB util=${gpu_util}%)"
echo "Training log: $train_log"
echo "Runner log: $runner_log"
