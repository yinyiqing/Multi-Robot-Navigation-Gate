#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export DRL_MULTI_EXPERIMENT_LABEL=D2
export DRL_MULTI_TRAIN_FILE_NAME="${DRL_MULTI_TRAIN_FILE_NAME:-independent_dense_actor_simple_td3_hparam_d2_s20260801}"
export DRL_MULTI_TRAINING_VERSION=dense-simple-td3-hparam-d2-controlled-ego-local-critic-v1
export DRL_MULTI_MAX_EPOCHS="${DRL_MULTI_MAX_EPOCHS:-4}"

# During Critic warmup, perturb exactly one active ego per joint step. The other
# four agents execute the frozen 5A policy without exploration noise.
export DRL_MULTI_RANDOM_LINEAR_EXPLORATION_STEPS="${DRL_MULTI_RANDOM_LINEAR_EXPLORATION_STEPS:-21000}"
export DRL_MULTI_RANDOM_LINEAR_EXPLORATION_SCOPE=single_ego
export DRL_MULTI_CRITIC_WARMUP_EXPL_NOISE=0.0
export DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS=1000000000

# The cooperative reward contains neighbor outcomes, so the training-only
# Critic receives ego-frame relative position and velocity context.
export DRL_MULTI_USE_LOCAL_CRITIC=1
export DRL_MULTI_LOCAL_CRITIC_GEOMETRY_ONLY=0
export DRL_MULTI_LOCAL_CRITIC_CONTEXT_MODE=ego_motion
export DRL_MULTI_LOCAL_CRITIC_MAX_AGENTS=5
export DRL_MULTI_CRITIC_INTERACTION_FRACTION=0.0

exec "$PROJECT_ROOT/scripts/start_training_dense_simple_td3_hparam_b.sh"
