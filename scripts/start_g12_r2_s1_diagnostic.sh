#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$PROJECT_ROOT/.g12_r2_s1_diagnostic.pid"
LOG_DIR="$PROJECT_ROOT/logs/active/capacity-wide-g12-r2/s1-diagnostic"
RUNNER_LOG="$LOG_DIR/runner.log"
ACTOR="$PROJECT_ROOT/TD3/pytorch_models/capacity_wide_r2_s0_broad_n1_seed20260811_best_actor.pth"
EXPECTED_ACTOR_SHA="7cb61925a4188e638859f88d38288e0431e5f05be489fa6107a77c7efaed3822"

declare -A EXPECTED_CASE_SHA=(
  ["$PROJECT_ROOT/experiments/02_课程学习/cases/stage1_single_local_cases.json"]="9cc79ec2a82908127c77fa00eff1448661814f8025176d769ccd4a03a8fb4b40"
  ["$PROJECT_ROOT/experiments/02_课程学习/cases/stage1e_single_rescue_cases.json"]="3b2566d8898d5380bc4d5295009d0b81e088bd96bec63939d5184d88a8cce4d9"
  ["$PROJECT_ROOT/experiments/02_课程学习/cases/stage1f_wall_parallel_rescue_cases.json"]="36906f6164d79551a09264f10e939779c68b8ca8ab366e78b50911f73974f563"
  ["$PROJECT_ROOT/experiments/02_课程学习/cases/stage1g_collision_guard_cases.json"]="d52dd8d1b5dd904ad7f4b8c55b60a258fc5cc4616469ead9932d93ee11be4403"
)

[[ -f "$ACTOR" ]] || { echo "S0 best Actor is missing: $ACTOR" >&2; exit 1; }
actual="$(sha256sum "$ACTOR" | awk '{print $1}')"
[[ "$actual" == "$EXPECTED_ACTOR_SHA" ]] || { echo "S0 best Actor hash mismatch" >&2; exit 1; }
for path in "${!EXPECTED_CASE_SHA[@]}"; do
  [[ -f "$path" ]] || { echo "Case file is missing: $path" >&2; exit 1; }
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "${EXPECTED_CASE_SHA[$path]}" ]] || { echo "Case hash mismatch: $path" >&2; exit 1; }
done

if [[ "${DRL_MULTI_DRY_RUN:-0}" == 1 ]]; then
  echo "G12-R2-S1 diagnostic dry run passed."
  echo "Actor: $ACTOR"
  echo "Episodes: 126 (42 cases x 3 repeats)"
  exit 0
fi

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "G12-R2-S1 diagnostic is already running with PID $old_pid" >&2
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

mkdir -p "$LOG_DIR"
setsid bash "$PROJECT_ROOT/scripts/run_g12_r2_s1_diagnostic_worker.sh" \
  >>"$RUNNER_LOG" 2>&1 < /dev/null &
echo $! > "$PID_FILE"

echo "Started G12-R2-S1 fixed-case diagnostic."
echo "PID: $(cat "$PID_FILE")"
echo "Episodes: 126 (42 cases x 3 repeats), serial"
echo "Estimated duration: 30-60 minutes"
echo "Runner log: $RUNNER_LOG"
