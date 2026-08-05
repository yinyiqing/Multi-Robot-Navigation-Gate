#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIEW_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/g11_a1_gate_v1"
D2_SUMMARY="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究/G11_D_Gate复核与独立准入/local_data/d2_summary.json"
MODEL_NAME="capacity_matched_actor_wide_n5_seed20260810_pilot"

[[ ! -f "$PROJECT_ROOT/.g11_d2_admission.pid" ]] || {
  echo "G11-D2 still owns its PID file; capacity training must wait for archival."
  exit 1
}
[[ -f "$D2_SUMMARY" ]] || {
  echo "G11-D2 archive summary is missing: $D2_SUMMARY"
  exit 1
}

declare -A EXPECTED_SHA256=(
  ["$VIEW_DIR/train.json.gz"]="a97534ed22d1b4b12951cd42b80d515037774830f38cb432926c0e9f50379026"
  ["$VIEW_DIR/validation.json.gz"]="e261a7afbac8f7341ab13609c2662a2824a0ff383789287ad7733290389cd99d"
  ["$PROJECT_ROOT/TD3/pytorch_models/TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best_actor.pth"]="fa28855049b67b3ee44c66d55d4f14441fc7c521e5429862c75b152f7d5cacc5"
)
for path in "${!EXPECTED_SHA256[@]}"; do
  [[ -f "$path" ]] || { echo "Required G12 input is missing: $path"; exit 1; }
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "${EXPECTED_SHA256[$path]}" ]] || {
    echo "G12 input hash mismatch: $path" >&2
    echo "expected=${EXPECTED_SHA256[$path]} actual=$actual" >&2
    exit 1
  }
done

(
  source "$PROJECT_ROOT/env.python.sh"
  export CUDA_VISIBLE_DEVICES=""
  python3 "$PROJECT_ROOT/scripts/audit_capacity_matched_actor.py"
)

export DRL_MULTI_TRAIN_MANIFEST="$VIEW_DIR/train.json.gz"
export DRL_MULTI_EVAL_MANIFEST="$VIEW_DIR/validation.json.gz"
export DRL_MULTI_EXPERIMENT_LABEL=G12-P1-capacity-matched-wide-actor
export DRL_MULTI_TRAIN_FILE_NAME="$MODEL_NAME"
export DRL_MULTI_TRAINING_VERSION=capacity-matched-wide-actor-pilot-v1
export DRL_MULTI_SEED=20260810
export DRL_MULTI_PID_FILE="$PROJECT_ROOT/.g12_capacity_actor.pid"
export DRL_MULTI_LOG_DIR="$PROJECT_ROOT/logs/active/capacity-matched-actor-g12-p1"

# 1,003,127 parameters, versus 1,003,604 in the two frozen Actor checkpoints.
export DRL_MULTI_ACTOR_HIDDEN_DIM_1=1137
export DRL_MULTI_ACTOR_HIDDEN_DIM_2=855
export DRL_MULTI_ALLOW_ACTOR_WARMSTART_EXPANSION=1

# One shared full Actor, original Critic, and the same individual navigation reward.
export DRL_MULTI_USE_LOCAL_CRITIC=0
export DRL_MULTI_USE_DYNAMIC_REWARD=0
export DRL_MULTI_PROGRESS_REWARD_WEIGHT=20.0
export DRL_MULTI_FORWARD_REWARD_WEIGHT=0.5
export DRL_MULTI_TURN_PENALTY_WEIGHT=0.2
export DRL_MULTI_OBSTACLE_PENALTY_WEIGHT=0.5
export DRL_MULTI_STAGNATION_PENALTY_WEIGHT=0.03
unset DRL_MULTI_TIMEOUT_REWARD

export DRL_MULTI_BATCH_SIZE=128
export DRL_MULTI_MIN_REPLAY_SIZE=5000
export DRL_MULTI_DISCOUNT=0.999
export DRL_MULTI_ACTOR_LR=0.000002
export DRL_MULTI_CRITIC_LR=0.00002
export DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS=20000
export DRL_MULTI_CRITIC_WARMUP_EXPL_NOISE=0.20
export DRL_MULTI_EXPL_NOISE=0.05
export DRL_MULTI_EXPL_MIN=0.02
export DRL_MULTI_EXPL_DECAY_STEPS=100000
export DRL_MULTI_RANDOM_LINEAR_EXPLORATION_STEPS=0
export DRL_MULTI_CONTROLLED_EGO_REPLAY_ONLY=0
export DRL_MULTI_CRITIC_INTERACTION_FRACTION=0.0

# Epoch 1 is the function-preserved 5A baseline; epochs 2-4 evaluate learning.
export DRL_MULTI_EVAL_FREQ_AGENT_SAMPLES=20000
export DRL_MULTI_EVAL_EPISODES=120
export DRL_MULTI_MAX_EPOCHS=4
export DRL_MULTI_EARLY_STOP_PATIENCE=1
export DRL_MULTI_EARLY_STOP_MIN_EPOCHS=2
export DRL_MULTI_EARLY_STOP_FULL_SUCCESS_DROP=0.10
export DRL_MULTI_EARLY_STOP_SUCCESS_DROP=0.08
export DRL_MULTI_EARLY_STOP_TIMEOUT_INCREASE=0.10
export DRL_MULTI_EARLY_STOP_TIMEOUT_ABSOLUTE=0.15

export DRL_MULTI_ROS_PORT="${DRL_MULTI_ROS_PORT:-14461}"
export DRL_MULTI_GAZEBO_PORT="${DRL_MULTI_GAZEBO_PORT:-14561}"

exec "$PROJECT_ROOT/scripts/start_training_dense_simple_td3_hparam_a.sh"
