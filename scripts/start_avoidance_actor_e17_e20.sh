#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD3_DIR="$ROOT/TD3"
VIEW_DIR="$ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/strong_interaction_curriculum_v1"
SOURCE_MODEL="interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726"
MODEL="avoidance_actor_from_5a_balanced_continue_e20_s20260813"
WEAK_MODEL="TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best"
SOURCE_CHECKPOINT="$TD3_DIR/checkpoints/${SOURCE_MODEL}_latest.pt"
CHECKPOINT="$TD3_DIR/checkpoints/${MODEL}_latest.pt"
TRAIN_MANIFEST="$VIEW_DIR/full_train.json.gz"
EVAL_MANIFEST="$VIEW_DIR/validation.json.gz"
LOG_DIR="$ROOT/logs/active/avoidance-actor-e17-e20"
PID_FILE="$ROOT/.avoidance_actor_e17_e20.pid"
LAUNCHFILE="multi_robot_scenario_strong_interaction_pilot_5.launch"
ROS_PORT="${DRL_MULTI_ROS_PORT:-13203}"
GAZEBO_PORT="${DRL_MULTI_GAZEBO_PORT:-13303}"

verify_sha() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    echo "SHA-256 mismatch: $path" >&2
    echo "expected=$expected actual=$actual" >&2
    exit 1
  }
}

verify_sha "$SOURCE_CHECKPOINT" "251b9c2efd61e7caf39e8534089c889e5708b7fc8cb55ea9ebe345fc31ef1788"
verify_sha "$TRAIN_MANIFEST" "d5b9b1fb968c8752e54e66f1ea3f25e7c2bf45eae3f012a686008704964da142"
verify_sha "$EVAL_MANIFEST" "3b2646a842b777f8c60dca4c452cb78eb3a223ffe59139b8501797aa1d23d583"
[[ -f "$TD3_DIR/pytorch_models/${WEAK_MODEL}_actor.pth" ]] || {
  echo "Missing frozen 5A Actor" >&2
  exit 1
}
[[ -f "$TD3_DIR/assets/$LAUNCHFILE" ]] || {
  echo "Missing launch file: $TD3_DIR/assets/$LAUNCHFILE" >&2
  exit 1
}

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' <"$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Avoidance Actor continuation already runs as PID $old_pid" >&2
    exit 1
  fi
  rm -f "$PID_FILE"
fi
for port in "$ROS_PORT" "$GAZEBO_PORT"; do
  if ss -ltn | awk '{print $4}' | grep -Eq ":${port}$"; then
    echo "Port $port is already in use" >&2
    exit 1
  fi
done

mkdir -p "$LOG_DIR" "$TD3_DIR/checkpoints"
if [[ ! -f "$CHECKPOINT" ]]; then
  cp --reflink=auto "$SOURCE_CHECKPOINT" "$CHECKPOINT"
fi

source "$ROOT/env.python.sh"
python3 - "$CHECKPOINT" <<'PY'
import sys, torch
s = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert s["epoch"] == 17, s["epoch"]
assert s["timestep"] == 320000, s["timestep"]
assert len(s["evaluations"]) == 16, len(s["evaluations"])
assert s["best_epoch"] == 16, s["best_epoch"]
assert len(s["train_seen_scenario_ids"]) == 2560
n = s["network"]
assert n["state_dim"] == 24 and n["critic_state_dim"] == 87
assert n["critic_context_mode"] == "ego_motion"
for key in ("actor", "actor_target", "actor_optimizer", "critic", "critic_target", "critic_optimizer"):
    assert key in n
print("Checkpoint audit passed: resume epoch 17 from 320000 samples")
PY

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$LOG_DIR/train_${MODEL}_${timestamp}.log"

setsid bash -lc "
  set -eo pipefail
  cleanup() {
    pgid=\"\$(ps -o pgid= -p \$\$ | tr -d ' ')\"
    ps -eo pid=,pgid= | awk -v pgid=\"\$pgid\" -v self=\"\$\$\" \
      '\$2 == pgid && \$1 != self { print \$1 }' | xargs -r kill 2>/dev/null || true
    rm -f '$PID_FILE'
  }
  trap cleanup EXIT
  source /opt/ros/noetic/setup.bash
  source '$ROOT/env.python.sh'
  source '$ROOT/catkin_ws/devel_isolated/setup.bash'
  export ROS_HOSTNAME=localhost
  export ROS_MASTER_URI=http://localhost:$ROS_PORT
  export ROS_PORT_SIM=$ROS_PORT
  export GAZEBO_MASTER_URI=http://localhost:$GAZEBO_PORT
  export GAZEBO_RESOURCE_PATH='$ROOT/catkin_ws/src/multi_robot_scenario/launch'
  export DRL_MULTI_NUM_AGENTS=5
  export DRL_MULTI_SEED=20260724
  export DRL_MULTI_TRAIN_LAUNCHFILE='$LAUNCHFILE'
  export DRL_MULTI_SCENARIO=manifest
  export DRL_MULTI_MANIFEST_PATH='$TRAIN_MANIFEST'
  export DRL_MULTI_EVAL_MANIFEST_PATH='$EVAL_MANIFEST'
  export DRL_MULTI_MANIFEST_SAMPLING=balanced_cycle
  export DRL_MULTI_TRAIN_FILE_NAME='$MODEL'
  export DRL_MULTI_RESUME_TRAINING=1
  export DRL_MULTI_MAX_EPOCHS=20
  export DRL_MULTI_EVAL_EPISODES=140
  export DRL_MULTI_EVAL_FREQ_AGENT_SAMPLES=20000
  export DRL_MULTI_BEST_METRIC=full_success
  export DRL_MULTI_TRAINING_VERSION='interaction-oracle-specialist-pilot-v6-focused-actor'
  export DRL_MULTI_ACTOR_TRAIN_MODE=full
  export DRL_MULTI_USE_ORACLE_INTERACTION_ROLLOUT=1
  export DRL_MULTI_ORACLE_WEAK_ACTOR_NAME='$WEAK_MODEL'
  export DRL_MULTI_ORACLE_INTERACTION_DISTANCE=2.0
  export DRL_MULTI_ACTOR_INTERACTION_ONLY=1
  export DRL_MULTI_USE_DYNAMIC_REWARD=1
  export DRL_MULTI_REWARD_MODE=average
  export DRL_MULTI_REWARD_SELF_WEIGHT=0.8
  export DRL_MULTI_USE_DISTANCE_WEIGHTED_REWARD=1
  export DRL_MULTI_REWARD_SIGMA=2.0
  export DRL_MULTI_INTERACTION_SAFE_DISTANCE=1.2
  export DRL_MULTI_INTERACTION_CLOSE_PENALTY=0.5
  export DRL_MULTI_INTERACTION_STAGNATION_PENALTY=0.05
  export DRL_MULTI_USE_SAFE_RECOVERY_REWARD=0
  export DRL_MULTI_USE_ANTI_STAGNATION_REWARD=0
  export DRL_MULTI_USE_LOCAL_NAVIGATION_REWARD=0
  export DRL_MULTI_USE_WALL_CLEARANCE_REWARD=0
  export DRL_MULTI_USE_YIELD_PRIORITY_REWARD=0
  export DRL_MULTI_FORWARD_REWARD_WEIGHT=0.0
  export DRL_MULTI_STAGNATION_PENALTY_WEIGHT=0.0
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
  export DRL_MULTI_CRITIC_INTERACTION_FRACTION=0.75
  export DRL_MULTI_USE_ACTOR_GRADIENT_GATE=1
  export DRL_MULTI_ACTOR_GRADIENT_SAFETY_DISTANCE=1.2
  export DRL_MULTI_ACTOR_GRADIENT_GATE_BATCH_SIZE=512
  export DRL_MULTI_ACTOR_GRADIENT_GATE_MIN_SAMPLES=32
  export DRL_MULTI_ACTOR_GRADIENT_MAX_LINEAR_POSITIVE_SHARE=0.9
  export DRL_MULTI_ACTOR_GRADIENT_MAX_ANGULAR_ONE_SIDED_SHARE=0.9
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
  export DRL_MULTI_ACTOR_ANCHOR_WEIGHT=0.0
  export DRL_MULTI_ACTOR_Q_NORMALIZATION_ALPHA=0.0
  export DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS=21000
  export DRL_MULTI_ACTOR_LR=0.000001
  export DRL_MULTI_CRITIC_LR=0.00008
  export DRL_MULTI_POLICY_FREQ=2
  export DRL_MULTI_BATCH_SIZE=40
  export DRL_MULTI_DISCOUNT=0.99999
  export DRL_MULTI_TAU=0.005
  export DRL_MULTI_POLICY_NOISE=0.2
  export DRL_MULTI_NOISE_CLIP=0.5
  export DRL_MULTI_EXPL_NOISE=0.08
  export DRL_MULTI_CRITIC_WARMUP_EXPL_NOISE=0.30
  export DRL_MULTI_EXPL_MIN=0.03
  export DRL_MULTI_EXPL_DECAY_STEPS=80000
  unset DRL_MULTI_FIXED_PHYSICS_STEP_SIZE DRL_MULTI_REQUIRE_FIXED_STEP_SERVICE
  cd '$TD3_DIR'
  exec python3 -u train_velodyne_td3_multi.py
" >"$log_file" 2>&1 < /dev/null &

echo $! >"$PID_FILE"
echo "Avoidance Actor epoch 17-20 continuation started"
echo "PID: $(cat "$PID_FILE")"
echo "Model: $MODEL"
echo "Log: $log_file"
