#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
RUN="$BASE/36_G34_G25slice_matched"
TEST="$RUN/local_data/matched"
MANIFEST="$BASE/25_最终消融与Sealed评测/local_data/sealed_manifest/dense_test_first256.json.gz"
CHECKPOINT="$BASE/34_双头RewardAwareGate/local_data/training/seed20260912/best_runtime.pt"
LOG_DIR="$RUN/logs/matched"
RESULT_DIR="$TEST/results"
STATE_DIR="$TEST/checkpoints"
PID_FILE="$ROOT/.g34_g25slice_matched.pid"
LAUNCHFILE="$LOG_DIR/runtime_g34_g25slice.launch"

EPISODES=256
SEEDS=(20260901 20260902 20260903)
if [[ "${1:-}" == "--pilot" ]]; then
  EPISODES=16
  SEEDS=(20260901)
fi

[[ -f "$MANIFEST" ]] || { echo "Missing G25 manifest: $MANIFEST" >&2; exit 1; }
[[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" == "1098d13d09af2a4850c2d227eec8fe87f05a7de3b3a8d830de2dd472a059e211" ]] || { echo "G25 manifest hash mismatch" >&2; exit 1; }
[[ -f "$CHECKPOINT" ]] || { echo "Missing G34 checkpoint: $CHECKPOINT" >&2; exit 1; }
[[ "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" == "4cb3626289c5b353d3ccf858f02c0e06d828106aeadc9f551f62ed6c90e4e95d" ]] || { echo "G34 checkpoint hash mismatch" >&2; exit 1; }

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' <"$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "G34 G25-slice matched evaluation already running as PID $old_pid"
    exit 0
  fi
  unlink "$PID_FILE"
fi

mkdir -p "$LOG_DIR" "$RESULT_DIR" "$STATE_DIR"
/usr/bin/python3 "$ROOT/scripts/generate_multi_robot_launch.py" --num-agents 5 --output "$LAUNCHFILE"
git -C "$ROOT" rev-parse HEAD >"$LOG_DIR/git_commit.txt"
sha256sum "$ROOT/TD3/learned_gate_controller.py" "$ROOT/TD3/reward_aware_gate.py" \
  "$ROOT/TD3/test_velodyne_td3_multi.py" "$ROOT/TD3/multi_agent_velodyne_env.py" \
  "$ROOT/scripts/run_g34_g25slice_matched_worker.sh" "$ROOT/scripts/analyze_g34_g25slice_matched.py" \
  "$MANIFEST" "$CHECKPOINT" >"$LOG_DIR/code_sha256.txt"

export G34_MATCHED_MANIFEST="$MANIFEST"
export G34_MATCHED_LOG_DIR="$LOG_DIR"
export G34_MATCHED_RESULT_DIR="$RESULT_DIR"
export G34_MATCHED_STATE_DIR="$STATE_DIR"
export G34_MATCHED_PID_FILE="$PID_FILE"
export G34_MATCHED_LAUNCHFILE="$LAUNCHFILE"
export G34_MATCHED_EPISODES="$EPISODES"
export G34_MATCHED_SEEDS="${SEEDS[*]}"
export G34_MATCHED_ROS_PORT=18652 G34_MATCHED_GAZEBO_PORT=19652

setsid bash "$ROOT/scripts/run_g34_g25slice_matched_worker.sh" >>"$LOG_DIR/runner.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" >"$PID_FILE"
echo "Started G34 G25-slice matched evaluation: ${#SEEDS[@]} repeat(s) x $EPISODES scenes = $(( ${#SEEDS[@]} * EPISODES )) episodes"
echo "PID: $pid"
echo "Live log: $LOG_DIR/runner.log"
