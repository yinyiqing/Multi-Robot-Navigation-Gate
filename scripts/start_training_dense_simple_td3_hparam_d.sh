#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export DRL_MULTI_EXPERIMENT_LABEL=D
export DRL_MULTI_TRAIN_FILE_NAME="${DRL_MULTI_TRAIN_FILE_NAME:-independent_dense_actor_simple_td3_hparam_d_s20260801}"
export DRL_MULTI_TRAINING_VERSION=dense-simple-td3-hparam-d-full-random-linear-v1
export DRL_MULTI_MAX_EPOCHS="${DRL_MULTI_MAX_EPOCHS:-4}"
export DRL_MULTI_RANDOM_LINEAR_EXPLORATION_STEPS="${DRL_MULTI_RANDOM_LINEAR_EXPLORATION_STEPS:-21000}"
export DRL_MULTI_CRITIC_WARMUP_EXPL_NOISE="${DRL_MULTI_CRITIC_WARMUP_EXPL_NOISE:-0.10}"

exec "$PROJECT_ROOT/scripts/start_training_dense_simple_td3_hparam_b.sh"
