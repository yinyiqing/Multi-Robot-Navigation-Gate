#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$ROOT/TD3"
VIEW_DIR="$ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/g12_full_scene_selection_v1"
MANIFEST="$VIEW_DIR/validation.json.gz"
FIVE_A="$TD3_DIR/pytorch_models/TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best_actor.pth"
OLD_ACTOR="$TD3_DIR/pytorch_models/interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016_actor.pth"
NEW_ACTOR="$TD3_DIR/pytorch_models/avoidance_actor_from_5a_balanced_continue_e20_s20260813_best_actor.pth"
LOG_DIR="$ROOT/logs/active/avoidance-actor-matched-admission"
PID_FILE="$ROOT/.avoidance_actor_matched_admission.pid"
ROS_PORT=15813
GAZEBO_PORT=15913

verify_sha() {
  local path="$1" expected="$2" actual
  [[ -f "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    echo "SHA-256 mismatch: $path" >&2
    echo "expected=$expected actual=$actual" >&2
    exit 1
  }
}

verify_sha "$MANIFEST" "52435d6c5bdf9914e7212dd29cb4bfec074257f72d85f0d71741deee7c63b635"
verify_sha "$FIVE_A" "fa28855049b67b3ee44c66d55d4f14441fc7c521e5429862c75b152f7d5cacc5"
verify_sha "$OLD_ACTOR" "6ec1942fcd497ab1cc2a85a5aaec8f524395dc21ff21a442dca243a52e917c0b"
verify_sha "$NEW_ACTOR" "149c2e42848ecc9bc478cbed7fd89b9062936dbd5c669b55e6964441685155a5"

if [[ "${DRL_MULTI_DRY_RUN:-0}" == 1 ]]; then
  echo "Avoidance Actor matched admission dry run passed"
  echo "Manifest: $MANIFEST"
  echo "Policies: 5A+epoch16, 5A+epoch17"
  echo "Seeds: 20260814 20260815; 120 episodes each"
  exit 0
fi

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' <"$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Matched admission already runs as PID $old_pid" >&2
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

mkdir -p "$LOG_DIR"
/usr/bin/python3 "$ROOT/scripts/generate_multi_robot_launch.py" \
  --num-agents 5 --output "$LOG_DIR/runtime_avoidance_actor_matched_admission.launch"
setsid bash "$ROOT/scripts/run_avoidance_actor_matched_admission_worker.sh" \
  >>"$LOG_DIR/runner.log" 2>&1 < /dev/null &
echo $! >"$PID_FILE"
echo "Avoidance Actor matched admission started"
echo "PID: $(cat "$PID_FILE")"
echo "Episodes: 480 total"
echo "Log: $LOG_DIR/runner.log"
