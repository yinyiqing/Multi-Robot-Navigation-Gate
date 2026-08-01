#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export DRL_MULTI_EXPERIMENT_LABEL=B
export DRL_MULTI_TRAIN_FILE_NAME="${DRL_MULTI_TRAIN_FILE_NAME:-independent_dense_actor_simple_td3_hparam_b_s20260801}"
export DRL_MULTI_TRAINING_VERSION=dense-simple-td3-hparam-b-v1
export DRL_MULTI_MAX_EPOCHS="${DRL_MULTI_MAX_EPOCHS:-4}"

# Keep the reward compact: navigation progress, clearance, and terminal outcomes.
export DRL_MULTI_PROGRESS_REWARD_WEIGHT=10.0
export DRL_MULTI_FORWARD_REWARD_WEIGHT=0.0
export DRL_MULTI_TURN_PENALTY_WEIGHT=0.05
export DRL_MULTI_OBSTACLE_PENALTY_WEIGHT=1.0
export DRL_MULTI_STAGNATION_PENALTY_WEIGHT=0.0
export DRL_MULTI_TIMEOUT_REWARD=-150.0

# Train the fresh Critic first, then fine-tune the warm-started Actor slowly.
export DRL_MULTI_BATCH_SIZE=128
export DRL_MULTI_MIN_REPLAY_SIZE=6000
export DRL_MULTI_DISCOUNT=0.999
export DRL_MULTI_ACTOR_LR=0.000002
export DRL_MULTI_CRITIC_LR=0.00002
export DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS=21000

# Broaden Critic warmup coverage without keeping high noise after Actor unlock.
export DRL_MULTI_EXPL_NOISE=0.05
export DRL_MULTI_CRITIC_WARMUP_EXPL_NOISE=0.25
export DRL_MULTI_EXPL_MIN=0.02
export DRL_MULTI_EXPL_DECAY_STEPS=100000

exec "$PROJECT_ROOT/scripts/start_training_dense_simple_td3_hparam_a.sh"
