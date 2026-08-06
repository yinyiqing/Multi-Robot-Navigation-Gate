#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIEW_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/g11_a1_gate_v1"
D2_SUMMARY="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究/G11_D_Gate复核与独立准入/local_data/d2_summary.json"
MODEL_NAME="capacity_original_width_r1_n5_seed20260810"

[[ ! -f "$PROJECT_ROOT/.g11_d2_admission.pid" ]] || {
  echo "G11-D2 still owns its PID file; R1 must wait for archival."
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
  [[ -f "$path" ]] || { echo "Required G12-R1 input is missing: $path"; exit 1; }
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "${EXPECTED_SHA256[$path]}" ]] || {
    echo "G12-R1 input hash mismatch: $path" >&2
    echo "expected=${EXPECTED_SHA256[$path]} actual=$actual" >&2
    exit 1
  }
done

command -v nvidia-smi >/dev/null 2>&1 || {
  echo "nvidia-smi is unavailable; G12-R1 requires CUDA."
  exit 1
}
gpu_state="$(nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | head -n 1)"
free_mib="$(printf '%s' "$gpu_state" | cut -d, -f1 | tr -d '[:space:]')"
utilization="$(printf '%s' "$gpu_state" | cut -d, -f2 | tr -d '[:space:]')"
if [[ ! "$free_mib" =~ ^[0-9]+$ || ! "$utilization" =~ ^[0-9]+$ ]] \
  || (( free_mib < 8192 || utilization > 20 )); then
  echo "GPU 0 is not ready for G12-R1 (free=${free_mib:-unknown} MiB, utilization=${utilization:-unknown}%)."
  exit 1
fi
export CUDA_VISIBLE_DEVICES=0

export DRL_MULTI_TRAIN_MANIFEST="$VIEW_DIR/train.json.gz"
export DRL_MULTI_EVAL_MANIFEST="$VIEW_DIR/validation.json.gz"
export DRL_MULTI_EXPERIMENT_LABEL=G12-R1-original-width-control
export DRL_MULTI_TRAIN_FILE_NAME="$MODEL_NAME"
export DRL_MULTI_TRAINING_VERSION=capacity-original-width-control-r1-v1
export DRL_MULTI_SEED=20260810
export DRL_MULTI_PID_FILE="$PROJECT_ROOT/.g12_capacity_r1.pid"
export DRL_MULTI_LOG_DIR="$PROJECT_ROOT/logs/active/capacity-original-width-r1"

# R1 changes only Actor width relative to P1 during the first 40k samples.
export DRL_MULTI_ACTOR_HIDDEN_DIM_1=800
export DRL_MULTI_ACTOR_HIDDEN_DIM_2=600
export DRL_MULTI_ALLOW_ACTOR_WARMSTART_EXPANSION=0

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

# Epoch 1 is the unchanged 5A boundary; Epoch 2 measures 20k Actor updates.
export DRL_MULTI_EVAL_FREQ_AGENT_SAMPLES=20000
export DRL_MULTI_EVAL_EPISODES=120
export DRL_MULTI_MAX_EPOCHS=2
export DRL_MULTI_EARLY_STOP_PATIENCE=1
export DRL_MULTI_EARLY_STOP_MIN_EPOCHS=2
export DRL_MULTI_EARLY_STOP_FULL_SUCCESS_DROP=0.10
export DRL_MULTI_EARLY_STOP_SUCCESS_DROP=0.08
export DRL_MULTI_EARLY_STOP_TIMEOUT_INCREASE=0.10
export DRL_MULTI_EARLY_STOP_TIMEOUT_ABSOLUTE=0.15

export DRL_MULTI_ROS_PORT="${DRL_MULTI_ROS_PORT:-14462}"
export DRL_MULTI_GAZEBO_PORT="${DRL_MULTI_GAZEBO_PORT:-14562}"

exec "$PROJECT_ROOT/scripts/start_training_dense_simple_td3_hparam_a.sh"
