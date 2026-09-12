#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
G34="$BASE/34_双头RewardAwareGate"
TEST="$G34/local_data/independent_test"
PROTOCOL="$TEST/protocol.json"
MANIFEST="$TEST/manifest/dense_test_640_896.json.gz"
RECORD="$TEST/manifest/record.json"
CHECKPOINT="$G34/local_data/training/seed20260912/best_runtime.pt"
CONFIRMATION="$G34/local_data/confirmation64/summary.json"
LOG_DIR="$G34/logs/independent_test"
RESULT_DIR="$TEST/results"
STATE_DIR="$TEST/checkpoints"
PID_FILE="$ROOT/.g34_independent_test.pid"
LAUNCHFILE="$LOG_DIR/runtime_g34_p3.launch"

verify_sha() {
  local path="$1" expected="$2"
  [[ -f "$path" ]] || { echo "Missing frozen input: $path" >&2; exit 1; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]] || {
    echo "SHA-256 mismatch: $path" >&2
    exit 1
  }
}

verify_sha "$ROOT/TD3/pytorch_models/TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best_actor.pth" "fa28855049b67b3ee44c66d55d4f14441fc7c521e5429862c75b152f7d5cacc5"
verify_sha "$ROOT/TD3/pytorch_models/interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016_actor.pth" "6ec1942fcd497ab1cc2a85a5aaec8f524395dc21ff21a442dca243a52e917c0b"
verify_sha "$BASE/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt" "0b914c0d090bbaba0a2be63c0d75d88580bf4d71778a7f1749c63989e3dbbd56"
verify_sha "$CHECKPOINT" "4cb3626289c5b353d3ccf858f02c0e06d828106aeadc9f551f62ed6c90e4e95d"
verify_sha "$CONFIRMATION" "d4efdc302bc8a54596737da153a2cd7c4603bcc625efe4e2f88284f386bd00dc"
verify_sha "$PROTOCOL" "f0c723798f6d8f747131c4733f483102040087a40212197a576d39e1de14701a"
verify_sha "$MANIFEST" "a0e5aa2884c75976503e34bda8316cfac3e99578317c501b53e4eba987e223cc"

python3 - "$RECORD" "$PROTOCOL" "$MANIFEST" <<'PY'
import hashlib,json,sys
from pathlib import Path
record_path,protocol_path,manifest_path=map(Path,sys.argv[1:])
record=json.loads(record_path.read_text(encoding='utf-8'))
if record.get('protocol_sha256') != hashlib.sha256(protocol_path.read_bytes()).hexdigest():
    raise SystemExit('manifest record/protocol mismatch')
if record.get('output_sha256') != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
    raise SystemExit('manifest record/hash mismatch')
if record.get('source_slice_zero_based') != [640,896] or record.get('scenes') != 256:
    raise SystemExit('manifest record/slice mismatch')
if any(v.get('scene_id_overlap') or v.get('complete_geometry_overlap') for v in record.get('overlap_audit',{}).values()):
    raise SystemExit('independent test overlap audit failed')
PY

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' <"$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "G34 independent test already running as PID $old_pid"
    exit 0
  fi
  unlink "$PID_FILE"
fi
[[ ! -f "$TEST/completion.json" ]] || { echo "G34 independent test already complete"; exit 0; }

mkdir -p "$LOG_DIR" "$RESULT_DIR" "$STATE_DIR"
/usr/bin/python3 "$ROOT/scripts/generate_multi_robot_launch.py" --num-agents 5 --output "$LAUNCHFILE"
git -C "$ROOT" rev-parse HEAD >"$LOG_DIR/git_commit.txt"
sha256sum "$ROOT/TD3/learned_gate_controller.py" "$ROOT/TD3/reward_aware_gate.py" \
  "$ROOT/TD3/test_velodyne_td3_multi.py" "$ROOT/TD3/multi_agent_velodyne_env.py" \
  "$ROOT/scripts/run_g34_independent_test_worker.sh" "$ROOT/scripts/analyze_g34_independent_test.py" \
  "$PROTOCOL" "$MANIFEST" >"$LOG_DIR/code_sha256.txt"

export G34_P3_MANIFEST="$MANIFEST" G34_P3_LOG_DIR="$LOG_DIR"
export G34_P3_RESULT_DIR="$RESULT_DIR" G34_P3_STATE_DIR="$STATE_DIR"
export G34_P3_PID_FILE="$PID_FILE" G34_P3_LAUNCHFILE="$LAUNCHFILE"
export G34_P3_ROS_PORT=18642 G34_P3_GAZEBO_PORT=19642
setsid bash "$ROOT/scripts/run_g34_independent_test_worker.sh" >>"$LOG_DIR/runner.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" >"$PID_FILE"
echo "Started G34 P3: 2 methods x 256 scenes x 3 repeats = 1536 episodes"
echo "PID: $pid"
echo "Live log: $LOG_DIR/runner.log"
