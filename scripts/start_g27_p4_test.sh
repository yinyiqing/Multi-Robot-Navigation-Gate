#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
G27="$BASE/27_反事实Reward增强Gate监督"
P3_SUMMARY="$G27/local_data/validation/p3_summary.json"
P3_RESULT="$G27/local_data/validation/results/g27_dense256_b3_s20260810.npy"
AMENDMENT="$G27/local_data/protocol/p4_continuation_amendment.json"
MANIFEST="$G27/local_data/test/manifests/dense_test_384_640.json.gz"
RECORD="$G27/local_data/test/manifests/manifest_record.json"
B3="$G27/local_data/training/seed20260910/best.pt"
LOG_DIR="$G27/logs/p4_test"
ARCHIVE_DIR="$G27/logs/archive/p4_test"
RESULT_DIR="$G27/local_data/test/results"
STATE_DIR="$G27/local_data/test/checkpoints"
COMPLETION="$G27/local_data/test/p4_completion.json"
PID_FILE="$ROOT/.g27_p4_test.pid"
LAUNCHFILE="$LOG_DIR/runtime_g27_p4.launch"

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
verify_sha "$BASE/11_可部署在线Gate研究/G11_B_student_rollout_v1/local_data/training/seed20260804/any/T1/best.pt" "fc59b4f783f7c5461ebb0239fab4b34896ad910ee78e7223e88d29ce9c3f5a52"

python3 - "$P3_SUMMARY" "$P3_RESULT" "$AMENDMENT" "$MANIFEST" "$RECORD" "$B3" <<'PY'
import hashlib,json,sys
from pathlib import Path

p3_path,p3_result_path,amendment_path,manifest_path,record_path,b3_path=map(Path,sys.argv[1:])
for path in (p3_path,p3_result_path,amendment_path,manifest_path,record_path,b3_path):
    if not path.is_file(): raise SystemExit('missing P4 prerequisite: %s' % path)
p3=json.loads(p3_path.read_text(encoding='utf-8'))
amendment=json.loads(amendment_path.read_text(encoding='utf-8'))
record=json.loads(record_path.read_text(encoding='utf-8'))
criteria=p3.get('admission_criteria',{})
required=('full_success_not_below_b2','robot_collision_increase_at_most_2pp','episode_timeout_increase_at_most_1pp')
if p3.get('admission_passed') is not False or not all(criteria.get(name) is True for name in required):
    raise SystemExit('P3 success/safety history does not authorize the amended continuation')
if criteria.get('interaction_share_down_10pp_or_paired_steps_down_5') is not False:
    raise SystemExit('P3 efficiency history does not match the amendment')
if amendment.get('decision') != 'continue_to_independent_test_with_frozen_b3':
    raise SystemExit('P4 continuation amendment is missing or invalid')
status=amendment.get('test_data_status_at_registration',{})
if status != {'p4_manifest_generated':False,'dense_test_slice_384_640_read':False,'p4_rollouts_started':False}:
    raise SystemExit('continuation was not registered before P4 test access')
manifest_hash=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
b3_hash=hashlib.sha256(b3_path.read_bytes()).hexdigest()
p3_hash=hashlib.sha256(p3_path.read_bytes()).hexdigest()
p3_result_hash=hashlib.sha256(p3_result_path.read_bytes()).hexdigest()
amendment_hash=hashlib.sha256(amendment_path.read_bytes()).hexdigest()
if record.get('output_sha256') != manifest_hash or record.get('scenes') != 256:
    raise SystemExit('P4 manifest record mismatch')
if record.get('start_index_in_source') != 384 or record.get('stop_index_in_source') != 640:
    raise SystemExit('P4 source slice mismatch')
if record.get('p3_summary_sha256') != p3_hash or amendment.get('p3_summary_sha256') != p3_hash:
    raise SystemExit('P4 record does not match the P3 summary')
if record.get('p3_result_sha256') != p3_result_hash or amendment.get('p3_result_sha256') != p3_result_hash:
    raise SystemExit('P4 authorization does not match the P3 result')
if record.get('continuation_amendment_sha256') != amendment_hash:
    raise SystemExit('P4 record does not match the continuation amendment')
if p3.get('b3_checkpoint_sha256') != b3_hash or record.get('b3_checkpoint_sha256') != b3_hash:
    raise SystemExit('B3 checkpoint hash mismatch')
if amendment.get('b3_checkpoint_sha256') != b3_hash:
    raise SystemExit('P4 amendment does not match the B3 checkpoint')
if any(values.get('scene_id_overlap') or values.get('complete_geometry_overlap') for values in record.get('overlap_audit',{}).values()):
    raise SystemExit('P4 overlap audit is not clean')
PY

[[ ! -e "$COMPLETION" ]] || { echo "P4 completion already exists: $COMPLETION" >&2; exit 1; }
[[ ! -e "$ARCHIVE_DIR" ]] || { echo "P4 archive already exists: $ARCHIVE_DIR" >&2; exit 1; }
if [[ -f "$PID_FILE" ]]; then
  pid="$(tr -d '[:space:]' <"$PID_FILE")"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && {
    echo "P4 is already active with PID $pid" >&2
    exit 1
  }
  unlink "$PID_FILE"
fi

mkdir -p "$LOG_DIR" "$RESULT_DIR" "$STATE_DIR"
/usr/bin/python3 "$ROOT/scripts/generate_multi_robot_launch.py" --num-agents 5 --output "$LAUNCHFILE"
git -C "$ROOT" rev-parse HEAD >"$LOG_DIR/git_commit.txt"
sha256sum \
  "$ROOT/TD3/learned_gate_controller.py" \
  "$ROOT/TD3/reward_aware_gate.py" \
  "$ROOT/TD3/test_velodyne_td3_multi.py" \
  "$ROOT/TD3/multi_agent_velodyne_env.py" \
  "$ROOT/scripts/run_g27_p4_test_worker.sh" \
  "$ROOT/scripts/analyze_g27_p4_test.py" \
  "$AMENDMENT" \
  >"$LOG_DIR/code_sha256.txt"

export G27_P4_MANIFEST="$MANIFEST" G27_P4_EPISODES=256
export G27_P4_AMENDMENT="$AMENDMENT"
export G27_P4_SEEDS="20260911 20260912 20260913"
export G27_P4_LOG_DIR="$LOG_DIR" G27_P4_ARCHIVE_DIR="$ARCHIVE_DIR"
export G27_P4_RESULT_DIR="$RESULT_DIR" G27_P4_STATE_DIR="$STATE_DIR"
export G27_P4_PID_FILE="$PID_FILE" G27_P4_LAUNCHFILE="$LAUNCHFILE"
export G27_P4_ROS_PORT=18520 G27_P4_GAZEBO_PORT=19520

setsid bash "$ROOT/scripts/run_g27_p4_test_worker.sh" >>"$LOG_DIR/runner.log" 2>&1 < /dev/null &
echo $! >"$PID_FILE"
echo "Started G27 P4: 3 methods x 256 scenes x 3 repeats"
echo "PID: $(tr -d '[:space:]' <"$PID_FILE")"
echo "Live log: $LOG_DIR/runner.log"
