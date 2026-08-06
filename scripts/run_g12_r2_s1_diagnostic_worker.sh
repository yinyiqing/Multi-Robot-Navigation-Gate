#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_NAME="capacity_wide_r2_s0_broad_n1_seed20260811_best"
RUN_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/12_参数匹配单Actor容量对照/local_data/s1_diagnostic"
RESULTS_DIR="$RUN_DIR/results"
CHECKPOINT_DIR="$RUN_DIR/checkpoints"
LOG_DIR="$PROJECT_ROOT/logs/active/capacity-wide-g12-r2/s1-diagnostic"
PID_FILE="$PROJECT_ROOT/.g12_r2_s1_diagnostic.pid"
LOCK_FILE="/tmp/local_critic_multi_robot_training.lock"
ROS_PORT=14621
GAZEBO_PORT=14721
source "$PROJECT_ROOT/scripts/lib_g12_r2_s1_runtime.sh"
STAGES=(
  stage1_single
  stage1e_single_rescue
  stage1f_wall_parallel_rescue
  stage1g_collision_guard
)

cleanup_stage() {
  local stage stage_pid_file stage_pid
  for stage in "${STAGES[@]}"; do
    stage_pid_file="$PROJECT_ROOT/.test_multi_curriculum_${stage}_detached.pid"
    [[ -f "$stage_pid_file" ]] || continue
    stage_pid="$(tr -d '[:space:]' < "$stage_pid_file")"
    g12_r2_s1_stop_stage "$stage_pid" "$ROS_PORT" "$GAZEBO_PORT" || true
    unlink "$stage_pid_file" 2>/dev/null || true
  done
}

cleanup() {
  cleanup_stage
  unlink "$PID_FILE" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

exec 9>"$LOCK_FILE"
flock -n 9 || { echo "Multi-robot lock is busy" >&2; exit 1; }
mkdir -p "$RESULTS_DIR" "$CHECKPOINT_DIR" "$LOG_DIR"

if ! g12_r2_s1_ports_are_free "$ROS_PORT" "$GAZEBO_PORT"; then
  echo "S1 ROS/Gazebo ports are occupied before the diagnostic starts" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0
export DRL_MULTI_SEED=20260812
export DRL_MULTI_TEST_ROS_PORT="$ROS_PORT"
export DRL_MULTI_TEST_GAZEBO_PORT="$GAZEBO_PORT"
export DRL_MULTI_TEST_LOG_DIR="$LOG_DIR"
export DRL_MULTI_TEST_ACTOR_MODE=full
export DRL_MULTI_ACTOR_SELECTION_MODE=single
export DRL_MULTI_CURRICULUM_SAMPLING=cycle
export DRL_MULTI_FIXED_PHYSICS_STEP_SIZE=0.001
export DRL_MULTI_REQUIRE_FIXED_STEP_SERVICE=1

for stage in "${STAGES[@]}"; do
  stats_path="$RESULTS_DIR/${stage}.npy"
  state_path="$CHECKPOINT_DIR/${stage}_state.pt"
  stage_pid_file="$PROJECT_ROOT/.test_multi_curriculum_${stage}_detached.pid"
  case "$stage" in
    stage1_single) target=18 ;;
    *) target=36 ;;
  esac

  if /usr/bin/python3 "$PROJECT_ROOT/scripts/analyze_g12_r2_s1_diagnostic.py" \
      --stage "$stage" --results-dir "$RESULTS_DIR" >/dev/null 2>&1; then
    echo "Skipping completed stage: $stage"
    continue
  fi
  rm -f "$stats_path" "$state_path"
  export DRL_MULTI_TEST_TARGET_EPISODES="$target"
  export DRL_MULTI_TEST_STATS_PATH="$stats_path"
  export DRL_MULTI_TEST_STATE_PATH="$state_path"
  export DRL_MULTI_TEST_LOG_TAG="g12_r2_s1_${stage}_seed20260812"
  echo "Starting $stage with $target episodes"
  bash "$PROJECT_ROOT/scripts/start_test_detached_multi_curriculum.sh" "$stage" "$MODEL_NAME"
  stage_pid="$(tr -d '[:space:]' < "$stage_pid_file")"
  while kill -0 "$stage_pid" 2>/dev/null; do
    sleep 5
  done
  g12_r2_s1_stop_stage "$stage_pid" "$ROS_PORT" "$GAZEBO_PORT"
  unlink "$stage_pid_file" 2>/dev/null || true
  /usr/bin/python3 "$PROJECT_ROOT/scripts/analyze_g12_r2_s1_diagnostic.py" \
    --stage "$stage" --results-dir "$RESULTS_DIR" >/dev/null
  echo "Completed $stage"
  sleep 1
done

/usr/bin/python3 "$PROJECT_ROOT/scripts/analyze_g12_r2_s1_diagnostic.py" \
  --results-dir "$RESULTS_DIR" --output "$RUN_DIR/summary.json"
echo "G12-R2-S1 diagnostic complete."
