#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$PROJECT_ROOT/TD3"
VIEW_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views"
N5_MANIFEST="${IE2_ADMISSION_MANIFEST:-$VIEW_DIR/g12_r2_curriculum_v1/n5/validation.json.gz}"
STRONG_TRAIN="${IE2_TRAIN_MANIFEST:-$VIEW_DIR/strong_interaction_curriculum_v1/full_train.json.gz}"
STRONG_VALIDATION="${IE2_EVAL_MANIFEST:-$VIEW_DIR/strong_interaction_curriculum_v1/validation.json.gz}"
ROUTE_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/15_E2恢复Actor诊断与训练"
RUN_DIR="${IE2_RUN_DIR:-$ROUTE_DIR/local_data/e2_ie2_overnight_pipeline}"
RESULTS_DIR="$RUN_DIR/results"
CHECKPOINT_DIR="$RUN_DIR/checkpoints"
TRAJECTORY_DIR="$RUN_DIR/trajectories"
LOG_DIR="${IE2_LOG_DIR:-$PROJECT_ROOT/logs/active/e2-ie2-overnight-pipeline}"
LAUNCHFILE="$LOG_DIR/runtime_e2_ie2_pipeline.launch"
PID_FILE="${IE2_PID_FILE:-$PROJECT_ROOT/.e2_ie2_overnight_pipeline.pid}"
LOCK_FILE="/tmp/local_critic_multi_robot_training.lock"
E2_MODEL="current_generalist_n5_efficiency_e2_s20260810_best"
OLD_INTERACTION_MODEL="interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016"
IE2_MODEL="${IE2_MODEL_NAME:-interaction_recovery_from_e2_strong40k_s20260820}"
OLD_RECOVERY_RESULT="$ROUTE_DIR/local_data/recovery_oracle_epoch16_pilot/results/e2_recovery_oracle_epoch16_pilot_s20260818.npy"
ROS_PORT=15657
GAZEBO_PORT=15757
SEED=20260818
TRAIN_SEED="${IE2_TRAIN_SEED:-20260820}"
EPISODES=120
TRAIN_SAMPLING="${IE2_TRAIN_SAMPLING:-balanced_cycle}"
TRAIN_EVAL_EPISODES="${IE2_TRAIN_EVAL_EPISODES:-140}"
TRAINING_VERSION="${IE2_TRAINING_VERSION:-e2-interaction-specialist-40k-pilot-v1}"
REWARD_MODE="${IE2_REWARD_MODE:-average}"
INTERACTION_STAGNATION_PENALTY="${IE2_INTERACTION_STAGNATION_PENALTY:-0.05}"
USE_SAFE_RECOVERY_REWARD="${IE2_USE_SAFE_RECOVERY_REWARD:-0}"
SAFE_RECOVERY_PENALTY="${IE2_SAFE_RECOVERY_PENALTY:-0.2}"
SAFE_RECOVERY_PROGRESS_BONUS_WEIGHT="${IE2_SAFE_RECOVERY_PROGRESS_BONUS_WEIGHT:-0.0}"
SAFE_RECOVERY_IDLE_PENALTY_WEIGHT="${IE2_SAFE_RECOVERY_IDLE_PENALTY_WEIGHT:-0.0}"

stop_runtime_children() {
  local pgid child_pids
  pgid="$(ps -o pgid= -p $$ | tr -d ' ')"
  child_pids="$(ps -eo pid=,pgid= | awk -v pgid="$pgid" -v self="$$" '$2 == pgid && $1 != self { print $1 }')"
  if [[ -n "$child_pids" ]]; then
    xargs -r kill -TERM 2>/dev/null <<<"$child_pids" || true
    sleep 3
    child_pids="$(ps -eo pid=,pgid= | awk -v pgid="$pgid" -v self="$$" '$2 == pgid && $1 != self { print $1 }')"
    [[ -z "$child_pids" ]] || xargs -r kill -KILL 2>/dev/null <<<"$child_pids" || true
  fi
  fuser -k -TERM "${ROS_PORT}/tcp" "${GAZEBO_PORT}/tcp" >/dev/null 2>&1 || true
}

cleanup() {
  stop_runtime_children
  unlink "$PID_FILE" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

exec 9>"$LOCK_FILE"
flock -n 9 || { echo "Multi-robot experiment lock is busy" >&2; exit 1; }
mkdir -p "$RESULTS_DIR" "$CHECKPOINT_DIR" "$TRAJECTORY_DIR" "$LOG_DIR"

set +u
source /opt/ros/noetic/setup.bash
source "$PROJECT_ROOT/env.python.sh"
source "$PROJECT_ROOT/catkin_ws/devel_isolated/setup.bash"
set -u

export CUDA_VISIBLE_DEVICES=0
export ROS_HOSTNAME=localhost
export GAZEBO_IP=127.0.0.1
export ROS_MASTER_URI="http://localhost:$ROS_PORT"
export ROS_PORT_SIM="$ROS_PORT"
export GAZEBO_MASTER_URI="http://localhost:$GAZEBO_PORT"
export GAZEBO_RESOURCE_PATH="$PROJECT_ROOT/catkin_ws/src/multi_robot_scenario/launch"

verify_test_result() {
  python3 - "$1" "$N5_MANIFEST" "$EPISODES" <<'PY'
import gzip, json, sys, numpy as np
rows = np.load(sys.argv[1], allow_pickle=True)
with gzip.open(sys.argv[2], "rt", encoding="utf-8") as handle:
    expected = [str(item["scenario_id"]) for item in json.load(handle)["scenarios"]]
observed = [str(row[12]) for row in rows]
if rows.shape != (int(sys.argv[3]), 17):
    raise SystemExit("wrong result shape: %s" % (rows.shape,))
if observed != expected or len(set(observed)) != len(observed):
    raise SystemExit("result scenario IDs do not match manifest order")
if sum(int(row[6]) + int(row[7]) + int(row[10]) for row in rows) != len(rows) * 5:
    raise SystemExit("terminal outcome accounting mismatch")
PY
}

wait_for_ports() {
  local port busy
  for _ in $(seq 1 60); do
    busy=0
    for port in "$ROS_PORT" "$GAZEBO_PORT"; do
      if ss -ltnH | awk '{print $4}' | grep -Eq ":${port}$"; then
        busy=1
      fi
    done
    [[ "$busy" -eq 0 ]] && return 0
    sleep 1
  done
  echo "ROS/Gazebo ports did not become free" >&2
  return 1
}

run_test() {
  local run_name="$1" standard_model="$2" interaction_model="$3" trajectory="$4"
  local state_path="$CHECKPOINT_DIR/${run_name}_state.pt"
  local stats_path="$RESULTS_DIR/${run_name}.npy"
  local log_file="$LOG_DIR/${run_name}.log"
  if [[ -f "$stats_path" ]] && verify_test_result "$stats_path" 2>/dev/null; then
    echo "Skipping completed test: $run_name"
    return 0
  fi
  for attempt in $(seq 1 3); do
    wait_for_ports
    set +e
    (
      export DRL_MULTI_NUM_AGENTS=5
      export DRL_MULTI_SEED="$SEED"
      export DRL_MULTI_TEST_LAUNCHFILE="$LAUNCHFILE"
      export DRL_MULTI_TEST_ACTOR_MODE=full
      export DRL_MULTI_TEST_TARGET_EPISODES="$EPISODES"
      export DRL_MULTI_SCENARIO=manifest
      export DRL_MULTI_MANIFEST_PATH="$N5_MANIFEST"
      export DRL_MULTI_MANIFEST_SAMPLING=cycle
      export DRL_MULTI_FIXED_PHYSICS_STEP_SIZE=0.001
      export DRL_MULTI_REQUIRE_FIXED_STEP_SERVICE=1
      export DRL_MULTI_STANDARD_ACTOR_FILE="$standard_model"
      export DRL_MULTI_TEST_FILE_NAME="$run_name"
      export DRL_MULTI_TEST_STATE_PATH="$state_path"
      export DRL_MULTI_TEST_STATS_PATH="$stats_path"
      if [[ -n "$interaction_model" ]]; then
        export DRL_MULTI_DENSE_ACTOR_FILE="$interaction_model"
        export DRL_MULTI_DENSE_ACTOR_MODE=full
        export DRL_MULTI_ACTOR_SELECTION_MODE=recovery_oracle
        export DRL_MULTI_RECOVERY_ORACLE_CANDIDATE_DISTANCE=2.0
        export DRL_MULTI_RECOVERY_ORACLE_RELEASE_DISTANCE=2.4
        export DRL_MULTI_RECOVERY_ORACLE_PROGRESS_THRESHOLD=0.003
        export DRL_MULTI_RECOVERY_ORACLE_PROGRESS_WINDOW=5
        export DRL_MULTI_RECOVERY_ORACLE_DISTANCE_DELTA_THRESHOLD=0.02
        export DRL_MULTI_RECOVERY_ORACLE_GOAL_DISTANCE=0.45
        export DRL_MULTI_RECOVERY_ORACLE_MINIMUM_HOLD_STEPS=3
        export DRL_MULTI_RECOVERY_ORACLE_MAXIMUM_HOLD_STEPS=20
      else
        export DRL_MULTI_ACTOR_SELECTION_MODE=single
        unset DRL_MULTI_DENSE_ACTOR_FILE
      fi
      if [[ -n "$trajectory" ]]; then
        export DRL_MULTI_TRAJECTORY_JSONL="$trajectory"
      else
        unset DRL_MULTI_TRAJECTORY_JSONL
      fi
      cd "$TD3_DIR"
      python3 -u test_velodyne_td3_multi.py
    ) >"${log_file%.log}_attempt${attempt}.log" 2>&1
    status=$?
    set -e
    stop_runtime_children
    wait_for_ports
    if verify_test_result "$stats_path" 2>/dev/null; then
      cp "${log_file%.log}_attempt${attempt}.log" "$log_file"
      echo "Completed test: $run_name"
      return 0
    fi
    echo "$run_name attempt $attempt incomplete (exit $status); retrying from saved state"
  done
  echo "$run_name did not complete after 3 attempts" >&2
  return 1
}

analyze_matched() {
  local new_result="${1:-}"
  local args=(
    --manifest "$N5_MANIFEST"
    --e2 "$RESULTS_DIR/e2_matched_control_s20260818.npy"
    --old-recovery "$OLD_RECOVERY_RESULT"
    --output "$RUN_DIR/summary.json"
    --seed "$SEED"
  )
  if [[ -n "$new_result" ]]; then
    args+=(--new-recovery "$new_result")
  fi
  python3 "$PROJECT_ROOT/scripts/analyze_e2_ie2_matched.py" "${args[@]}" \
    >"$LOG_DIR/analysis.log" 2>&1
}

verify_training_complete() {
  python3 - "$TD3_DIR/checkpoints/${IE2_MODEL}_latest.pt" <<'PY'
import sys, torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if len(state.get("evaluations", [])) < 2:
    raise SystemExit("training has fewer than two completed evaluations")
PY
  [[ -f "$TD3_DIR/pytorch_models/${IE2_MODEL}_best_actor.pth" ]]
}

run_ie2_training() {
  local attempt resume log_file status
  if verify_training_complete 2>/dev/null; then
    echo "Skipping completed I-E2 training"
    return 0
  fi
  for attempt in $(seq 1 3); do
    resume=0
    [[ -f "$TD3_DIR/checkpoints/${IE2_MODEL}_latest.pt" ]] && resume=1
    log_file="$LOG_DIR/train_${IE2_MODEL}_attempt${attempt}.log"
    wait_for_ports
    set +e
    (
      export DRL_MULTI_NUM_AGENTS=5
      export DRL_MULTI_SEED="$TRAIN_SEED"
      export DRL_MULTI_TRAIN_LAUNCHFILE="$LAUNCHFILE"
      export DRL_MULTI_SCENARIO=manifest
      export DRL_MULTI_MANIFEST_PATH="$STRONG_TRAIN"
      export DRL_MULTI_EVAL_MANIFEST_PATH="$STRONG_VALIDATION"
      export DRL_MULTI_MANIFEST_SAMPLING="$TRAIN_SAMPLING"
      export DRL_MULTI_TRAIN_FILE_NAME="$IE2_MODEL"
      export DRL_MULTI_TRAINING_VERSION="$TRAINING_VERSION"
      export DRL_MULTI_LOAD_MODEL=1
      export DRL_MULTI_LOAD_ACTOR_ONLY=1
      export DRL_MULTI_REQUIRE_MODEL_LOAD=1
      export DRL_MULTI_LOAD_MODEL_NAME="$E2_MODEL"
      export DRL_MULTI_ORACLE_WEAK_ACTOR_NAME="$E2_MODEL"
      export DRL_MULTI_RESUME_TRAINING="$resume"
      export DRL_MULTI_MAX_EPOCHS=2
      export DRL_MULTI_EVAL_EPISODES="$TRAIN_EVAL_EPISODES"
      export DRL_MULTI_EVAL_FREQ_AGENT_SAMPLES=20000
      export DRL_MULTI_BEST_METRIC=full_success
      export DRL_MULTI_ACTOR_TRAIN_MODE=full
      export DRL_MULTI_USE_DYNAMIC_REWARD=1
      export DRL_MULTI_REWARD_MODE="$REWARD_MODE"
      export DRL_MULTI_REWARD_SELF_WEIGHT=0.8
      export DRL_MULTI_USE_DISTANCE_WEIGHTED_REWARD=1
      export DRL_MULTI_REWARD_SIGMA=2.0
      export DRL_MULTI_INTERACTION_SAFE_DISTANCE=1.2
      export DRL_MULTI_INTERACTION_CLOSE_PENALTY=0.5
      export DRL_MULTI_INTERACTION_STAGNATION_PENALTY="$INTERACTION_STAGNATION_PENALTY"
      export DRL_MULTI_PROGRESS_REWARD_WEIGHT=20.0
      export DRL_MULTI_FORWARD_REWARD_WEIGHT=0.0
      export DRL_MULTI_STAGNATION_PENALTY_WEIGHT=0.0
      export DRL_MULTI_USE_SAFE_RECOVERY_REWARD="$USE_SAFE_RECOVERY_REWARD"
      export DRL_MULTI_SAFE_RECOVERY_PENALTY="$SAFE_RECOVERY_PENALTY"
      export DRL_MULTI_SAFE_RECOVERY_LINEAR_THRESHOLD=0.25
      export DRL_MULTI_SAFE_RECOVERY_PROGRESS_THRESHOLD=0.003
      export DRL_MULTI_SAFE_RECOVERY_MIN_LASER=0.6
      export DRL_MULTI_SAFE_RECOVERY_ROBOT_DISTANCE=1.2
      export DRL_MULTI_SAFE_RECOVERY_PROGRESS_BONUS_WEIGHT="$SAFE_RECOVERY_PROGRESS_BONUS_WEIGHT"
      export DRL_MULTI_SAFE_RECOVERY_IDLE_PENALTY_WEIGHT="$SAFE_RECOVERY_IDLE_PENALTY_WEIGHT"
      export DRL_MULTI_USE_ANTI_STAGNATION_REWARD=0
      export DRL_MULTI_USE_LOCAL_NAVIGATION_REWARD=0
      export DRL_MULTI_USE_WALL_CLEARANCE_REWARD=0
      export DRL_MULTI_ROBOT_SAFE_DISTANCE=1.2
      export DRL_MULTI_ROBOT_PROXIMITY_PENALTY_WEIGHT=5.0
      export DRL_MULTI_ROBOT_PROXIMITY_SPEED_PENALTY_WEIGHT=10.0
      export DRL_MULTI_ROBOT_CLEARANCE_REWARD_WEIGHT=20.0
      export DRL_MULTI_ROBOT_CLEARANCE_REWARD_MAX_GAIN=0.1
      export DRL_MULTI_USE_LOCAL_CRITIC=1
      export DRL_MULTI_LOCAL_CRITIC_GEOMETRY_ONLY=0
      export DRL_MULTI_LOCAL_CRITIC_CONTEXT_MODE=ego_motion
      export DRL_MULTI_LOCAL_CRITIC_MAX_AGENTS=10
      export DRL_MULTI_ACTIVE_NEIGHBORS_ONLY=1
      export DRL_MULTI_USE_ORACLE_INTERACTION_ROLLOUT=1
      export DRL_MULTI_USE_ORACLE_TARGET_POLICY=1
      export DRL_MULTI_ORACLE_INTERACTION_DISTANCE=2.0
      export DRL_MULTI_ACTOR_INTERACTION_ONLY=1
      export DRL_MULTI_CRITIC_INTERACTION_FRACTION=0.75
      export DRL_MULTI_USE_ACTOR_GRADIENT_GATE=1
      export DRL_MULTI_ACTOR_GRADIENT_SAFETY_DISTANCE=1.2
      export DRL_MULTI_ACTOR_GRADIENT_GATE_BATCH_SIZE=512
      export DRL_MULTI_ACTOR_GRADIENT_GATE_MIN_SAMPLES=32
      export DRL_MULTI_CRITIC_SAFETY_RANKING_WEIGHT=5.0
      export DRL_MULTI_CRITIC_SAFETY_RANKING_DISTANCE=1.0
      export DRL_MULTI_CRITIC_SAFETY_RANKING_MIN_CLOSING_SPEED=0.1
      export DRL_MULTI_CRITIC_SAFETY_RANKING_LINEAR_DELTA=0.4
      export DRL_MULTI_CRITIC_SAFETY_RANKING_MARGIN=0.1
      export DRL_MULTI_ACTOR_SAFETY_FOCUSED=1
      export DRL_MULTI_ACTOR_SAFETY_CANDIDATE_BATCH_SIZE=256
      export DRL_MULTI_ACTOR_SAFETY_MIN_SAMPLES=16
      export DRL_MULTI_ACTOR_SAFETY_DISTANCE=1.0
      export DRL_MULTI_ACTOR_SAFETY_MIN_CLOSING_SPEED=0.1
      export DRL_MULTI_ACTOR_ANGULAR_ANCHOR_WEIGHT=2.0
      export DRL_MULTI_ACTOR_LR=0.000001
      export DRL_MULTI_CRITIC_LR=0.00008
      export DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS=21000
      export DRL_MULTI_POLICY_FREQ=2
      export DRL_MULTI_ACTOR_ANCHOR_WEIGHT=0.0
      export DRL_MULTI_ACTOR_Q_NORMALIZATION_ALPHA=0.0
      export DRL_MULTI_BATCH_SIZE=40
      export DRL_MULTI_MIN_REPLAY_SIZE=0
      export DRL_MULTI_DISCOUNT=0.99999
      export DRL_MULTI_EXPL_NOISE=0.08
      export DRL_MULTI_CRITIC_WARMUP_EXPL_NOISE=0.3
      export DRL_MULTI_EXPL_MIN=0.03
      export DRL_MULTI_EXPL_DECAY_STEPS=80000
      export DRL_MULTI_FIXED_PHYSICS_STEP_SIZE=0.001
      export DRL_MULTI_REQUIRE_FIXED_STEP_SERVICE=1
      cd "$TD3_DIR"
      python3 -u train_velodyne_td3_multi.py
    ) >"$log_file" 2>&1
    status=$?
    set -e
    stop_runtime_children
    wait_for_ports
    if verify_training_complete 2>/dev/null; then
      echo "Completed I-E2 40k pilot"
      return 0
    fi
    echo "I-E2 training attempt $attempt incomplete (exit $status); resume=$resume"
  done
  echo "I-E2 training did not complete after 3 attempts" >&2
  return 1
}

echo "Stage 1/4: matched E2-only control"
run_test "e2_matched_control_s20260818" "$E2_MODEL" "" ""
echo "Stage 2/4: matched old-epoch16 analysis"
analyze_matched
echo "Stage 3/4: E2-based interaction Actor 40k pilot"
run_ie2_training
echo "Stage 4/4: E2 + I-E2 recovery-oracle evaluation"
NEW_RESULT="$RESULTS_DIR/e2_ie2_recovery_s20260818.npy"
run_test \
  "e2_ie2_recovery_s20260818" \
  "$E2_MODEL" \
  "${IE2_MODEL}_best" \
  "$TRAJECTORY_DIR/e2_ie2_recovery_s20260818.jsonl"
analyze_matched "$NEW_RESULT"
echo "E2 -> I-E2 overnight pipeline complete"
