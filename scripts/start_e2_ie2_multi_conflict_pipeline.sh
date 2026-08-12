#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
VIEW_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views"
ROUTE_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/15_E2恢复Actor诊断与训练"
TRAIN="$VIEW_DIR/ie2_multi_conflict_v1/train.json.gz"
VALIDATION="$VIEW_DIR/ie2_multi_conflict_v1/validation.json.gz"
N5_VALIDATION="$VIEW_DIR/g12_r2_curriculum_v1/n5/validation.json.gz"
E2_ACTOR="$TD3_DIR/pytorch_models/current_generalist_n5_efficiency_e2_s20260810_best_actor.pth"
OLD_INTERACTION_ACTOR="$TD3_DIR/pytorch_models/interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016_actor.pth"
PID_FILE="$PROJECT_ROOT/.e2_ie2_multi_conflict_pipeline.pid"
LOG_DIR="$PROJECT_ROOT/logs/active/e2-ie2-multi-conflict-pipeline"
RUNNER_LOG="$LOG_DIR/runner.log"
MODEL_NAME="interaction_recovery_from_e2_multi40k_s20260821"
ROS_PORT=15657
GAZEBO_PORT=15757

declare -A EXPECTED_SHA256=(
  ["$TRAIN"]="3c0be8a32513bde471cc5c6731f528397195fdda670be8e3bcf8cc7265ecaba8"
  ["$VALIDATION"]="9f44ee3962edb5f0750b8791cb357d0df09d3161552a94fb183a9b193cb6b5e0"
  ["$N5_VALIDATION"]="e33dbfad3d166fa4500b5997902a94c49108c77c646c7a39c26480b5054daef7"
  ["$E2_ACTOR"]="dbfd69a27aafae5cbb4686018bc074d3684de45ff397f1dffe77f5265138c7d0"
  ["$OLD_INTERACTION_ACTOR"]="6ec1942fcd497ab1cc2a85a5aaec8f524395dc21ff21a442dca243a52e917c0b"
)
for path in "${!EXPECTED_SHA256[@]}"; do
  [[ -f "$path" ]] || { echo "Pipeline input is missing: $path" >&2; exit 1; }
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "${EXPECTED_SHA256[$path]}" ]] || {
    echo "Pipeline input hash mismatch: $path" >&2
    echo "expected=${EXPECTED_SHA256[$path]} actual=$actual" >&2
    exit 1
  }
done

if [[ "${DRL_MULTI_DRY_RUN:-0}" == 1 ]]; then
  echo "I-E2-M multi-conflict pipeline dry run passed."
  echo "Train: 2400 cases, edge1/edge2/edge3+=960/720/720"
  echo "Internal validation: 200 cases, edge1/edge2/edge3+=140/30/30"
  echo "Pilot: fresh E2 warm start, 2 x 20k samples, Actor unlock at 21k"
  echo "Final matched admission: frozen N5 120 cases, seed 20260818"
  echo "Logs: $LOG_DIR"
  exit 0
fi

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "I-E2-M pipeline is already running with PID $old_pid" >&2
    exit 1
  fi
  unlink "$PID_FILE"
fi
if pgrep -af '^python3(\.8)? -u (train|test)_velodyne_td3_multi\.py($| )' >/dev/null; then
  echo "Another multi-robot training or evaluation process is running" >&2
  exit 1
fi
if pgrep -af '(^|/)(gzserver|rosmaster)( |$)' >/dev/null; then
  echo "An existing Gazebo or ROS master is running" >&2
  exit 1
fi
if ! flock -n /tmp/local_critic_multi_robot_training.lock -c true; then
  echo "Another multi-robot process holds the experiment lock" >&2
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
source "$PROJECT_ROOT/env.python.sh"
/usr/bin/python3 "$PROJECT_ROOT/scripts/generate_multi_robot_launch.py" \
  --num-agents 5 --output "$LOG_DIR/runtime_e2_ie2_pipeline.launch"

IE2_TRAIN_MANIFEST="$TRAIN" \
IE2_EVAL_MANIFEST="$VALIDATION" \
IE2_ADMISSION_MANIFEST="$N5_VALIDATION" \
IE2_RUN_DIR="$ROUTE_DIR/local_data/e2_ie2_multi_conflict_pipeline" \
IE2_LOG_DIR="$LOG_DIR" \
IE2_PID_FILE="$PID_FILE" \
IE2_MODEL_NAME="$MODEL_NAME" \
IE2_TRAIN_SEED=20260821 \
IE2_TRAIN_SAMPLING=cycle \
IE2_TRAIN_EVAL_EPISODES=200 \
IE2_TRAINING_VERSION=e2-interaction-specialist-multi-conflict-40k-pilot-v1 \
IE2_REWARD_MODE=average_plus_interaction \
IE2_INTERACTION_STAGNATION_PENALTY=0.02 \
IE2_USE_SAFE_RECOVERY_REWARD=1 \
IE2_SAFE_RECOVERY_PENALTY=0.10 \
IE2_SAFE_RECOVERY_PROGRESS_BONUS_WEIGHT=0.08 \
IE2_SAFE_RECOVERY_IDLE_PENALTY_WEIGHT=0.08 \
setsid bash "$PROJECT_ROOT/scripts/run_e2_ie2_overnight_pipeline_worker.sh" \
  >>"$RUNNER_LOG" 2>&1 < /dev/null &
echo $! > "$PID_FILE"

echo "Started I-E2-M multi-conflict pipeline."
echo "PID: $(cat "$PID_FILE")"
echo "GPU: 0 (free=${gpu_free_mib}MiB util=${gpu_util}%)"
echo "Runner log: $RUNNER_LOG"
