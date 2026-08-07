#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_NAME="capacity_wide_r2_s1_repair_n1_seed20260813"
ACTOR="$PROJECT_ROOT/TD3/pytorch_models/${MODEL_NAME}_actor.pth"
MANIFEST="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/g12_r2_curriculum_v1/n1/validation.json.gz"
PID_FILE="$PROJECT_ROOT/.g12_r2_s1_repair_validation.pid"
LOG_DIR="$PROJECT_ROOT/logs/active/capacity-wide-g12-r2/s1-repair"
RUNNER_LOG="$LOG_DIR/broad_validation_runner.log"
RUN_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/12_参数匹配单Actor容量对照/local_data/s1_repair_validation"
ROS_PORT=14641
GAZEBO_PORT=14741

declare -A EXPECTED_SHA256=(
  ["$ACTOR"]="e475be67d92d48277911731b4160afdbe21d25ceeba2f50160dbfe9734d18fe5"
  ["$MANIFEST"]="9ab4c5913f683d01e3ab186ea591d373abe1e835180f4a0bfeb469990269b125"
)
for path in "${!EXPECTED_SHA256[@]}"; do
  [[ -f "$path" ]] || { echo "Required validation input is missing: $path" >&2; exit 1; }
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "${EXPECTED_SHA256[$path]}" ]] || {
    echo "Validation input hash mismatch: $path" >&2
    exit 1
  }
done

if [[ "${DRL_MULTI_DRY_RUN:-0}" == 1 ]]; then
  echo "G12-R2-S1 broad validation dry run passed."
  echo "Actor: $ACTOR"
  echo "Manifest: $MANIFEST"
  echo "Episodes: 120"
  exit 0
fi

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "G12-R2-S1 broad validation is already running with PID $old_pid" >&2
    exit 1
  fi
  unlink "$PID_FILE"
fi
if pgrep -af '^python3(\.8)? -u (train|test)_velodyne_td3_multi.py($| )' >/dev/null; then
  echo "Another multi-robot training or evaluation process is running" >&2
  exit 1
fi
if pgrep -af '(^|/)(gzserver|rosmaster)( |$)' >/dev/null; then
  echo "An existing Gazebo or ROS master is running" >&2
  exit 1
fi
for port in "$ROS_PORT" "$GAZEBO_PORT"; do
  if ss -ltnH | awk '{print $4}' | grep -Eq ":${port}$"; then
    echo "Port $port is already in use" >&2
    exit 1
  fi
done
for output in "$RUN_DIR/results.npy" "$RUN_DIR/state.pt" "$RUN_DIR/summary.json"; do
  [[ ! -e "$output" ]] || { echo "Fresh validation output already exists: $output" >&2; exit 1; }
done

mkdir -p "$LOG_DIR" "$RUN_DIR"
timestamp="$(date +%Y%m%d_%H%M%S)"
validation_log="$LOG_DIR/broad_validation_${MODEL_NAME}_${timestamp}.log"
setsid bash "$PROJECT_ROOT/scripts/run_g12_r2_s1_repair_validation_worker.sh" \
  "$validation_log" >>"$RUNNER_LOG" 2>&1 < /dev/null &
echo $! > "$PID_FILE"

echo "Started G12-R2-S1 broad validation."
echo "PID: $(cat "$PID_FILE")"
echo "Episodes: 120 fixed n1 scenarios"
echo "Admission: full success >=117, collision <=3, timeout <=3"
echo "Validation log: $validation_log"
echo "Runner log: $RUNNER_LOG"
