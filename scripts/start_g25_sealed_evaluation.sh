#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/experiments/03_保留专门化/02_论文主线"
G25="$BASE/25_最终消融与Sealed评测/local_data"
MANIFEST="$G25/sealed_manifest/dense_test_first256.json.gz"
RECORD="$G25/sealed_manifest/manifest_record.json"
DRY_RUN="$G25/dry_run/dry_run_completion.json"
LOG_DIR="$ROOT/logs/active/g25-sealed-evaluation"
ARCHIVE_DIR="$ROOT/logs/archive/test/g25_sealed_evaluation"
PID_FILE="$ROOT/.g25_sealed_evaluation.pid"

verify_sha() {
  local path="$1" expected="$2"
  [[ -f "$path" ]] || { echo "Missing frozen input: $path" >&2; exit 1; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]] || {
    echo "SHA-256 mismatch: $path" >&2; exit 1
  }
}

verify_sha "$ROOT/TD3/pytorch_models/TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best_actor.pth" "fa28855049b67b3ee44c66d55d4f14441fc7c521e5429862c75b152f7d5cacc5"
verify_sha "$ROOT/TD3/pytorch_models/interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016_actor.pth" "6ec1942fcd497ab1cc2a85a5aaec8f524395dc21ff21a442dca243a52e917c0b"
verify_sha "$ROOT/TD3/pytorch_models/capacity_wide_r2b_5a_recipe_n5_seed20260823_best_actor.pth" "da28dd5820d09845eea07cb68da45a7afd262fe56e8a71f80bf6b5781551523a"
verify_sha "$BASE/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt" "0b914c0d090bbaba0a2be63c0d75d88580bf4d71778a7f1749c63989e3dbbd56"
verify_sha "$BASE/11_可部署在线Gate研究/G11_B_student_rollout_v1/local_data/training/seed20260804/any/T1/best.pt" "fc59b4f783f7c5461ebb0239fab4b34896ad910ee78e7223e88d29ce9c3f5a52"
verify_sha "$G25/ttc_rule/selection.json" "9bfb0a1a282dbd5f415c70326369bd54a3ebd6908a9db9fc371d62f5befa481a"

python3 - "$DRY_RUN" "$MANIFEST" "$RECORD" <<'PY'
import hashlib,json,sys
from pathlib import Path
dry_path,manifest_path,record_path=map(Path,sys.argv[1:])
for path in (dry_path,manifest_path,record_path):
    if not path.is_file(): raise SystemExit('missing sealed prerequisite: %s' % path)
dry=json.loads(dry_path.read_text())
record=json.loads(record_path.read_text())
if dry.get('status')!='complete' or dry.get('sealed_test_read'):
    raise SystemExit('dry-run prerequisite is invalid')
actual=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
if record.get('output_sha256')!=actual or record.get('scenes')!=256:
    raise SystemExit('sealed manifest record mismatch')
PY

git -C "$ROOT" diff --quiet -- . ':!trash/**' || { echo "Tracked worktree changes must be committed before sealed evaluation" >&2; exit 1; }
git -C "$ROOT" diff --cached --quiet -- . ':!trash/**' || { echo "Staged changes must be committed before sealed evaluation" >&2; exit 1; }
[[ ! -e "$ARCHIVE_DIR" ]] || { echo "Sealed archive already exists: $ARCHIVE_DIR" >&2; exit 1; }
if [[ -f "$PID_FILE" ]]; then
  pid="$(tr -d '[:space:]' <"$PID_FILE")"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && { echo "Sealed evaluation already active" >&2; exit 1; }
  unlink "$PID_FILE"
fi

mkdir -p "$LOG_DIR" "$G25/sealed/results" "$G25/sealed/checkpoints"
/usr/bin/python3 "$ROOT/scripts/generate_multi_robot_launch.py" --num-agents 5 --output "$LOG_DIR/runtime_sealed.launch"
git -C "$ROOT" rev-parse HEAD >"$LOG_DIR/git_commit.txt"
export G25_EVAL_PROFILE=sealed G25_EVAL_MANIFEST="$MANIFEST" G25_EVAL_EPISODES=256
export G25_EVAL_SEEDS="20260901 20260902 20260903"
export G25_EVAL_LOG_DIR="$LOG_DIR" G25_EVAL_ARCHIVE_DIR="$ARCHIVE_DIR"
export G25_EVAL_RESULT_DIR="$G25/sealed/results" G25_EVAL_STATE_DIR="$G25/sealed/checkpoints"
export G25_EVAL_PID_FILE="$PID_FILE" G25_EVAL_ROS_PORT=18423 G25_EVAL_GAZEBO_PORT=18523
setsid bash "$ROOT/scripts/run_g25_frozen_evaluation_worker.sh" >>"$LOG_DIR/runner.log" 2>&1 < /dev/null &
echo $! >"$PID_FILE"
echo "G25 sealed evaluation started: 7 methods x 256 scenes x 3 repeats"
echo "PID: $(cat "$PID_FILE")"
echo "Live log: $LOG_DIR/runner.log"
