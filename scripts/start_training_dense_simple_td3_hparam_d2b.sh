#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export DRL_MULTI_EXPERIMENT_LABEL=D2b
export DRL_MULTI_TRAIN_FILE_NAME="${DRL_MULTI_TRAIN_FILE_NAME:-independent_dense_actor_simple_td3_hparam_d2b_s20260801}"
export DRL_MULTI_TRAINING_VERSION=dense-simple-td3-hparam-d2b-controlled-ego-replay-v1

# One replay transition and one Critic update per joint environment step.
export DRL_MULTI_CONTROLLED_EGO_REPLAY_ONLY=1
export DRL_MULTI_RANDOM_LINEAR_EXPLORATION_SCOPE=single_ego
export DRL_MULTI_RANDOM_LINEAR_EXPLORATION_STEPS=13000
export DRL_MULTI_CRITIC_WARMUP_EXPL_NOISE=0.0
export DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS=1000000000

# A single 12k-sample run gives 9k post-warmup updates and validates only once.
export DRL_MULTI_MIN_REPLAY_SIZE=3000
export DRL_MULTI_EVAL_FREQ_AGENT_SAMPLES=12000
export DRL_MULTI_EVAL_EPISODES=50
export DRL_MULTI_MAX_EPOCHS=1

# Keep the D2 model, reward, and optimizer fixed.
export DRL_MULTI_CRITIC_LR=0.00002
export DRL_MULTI_USE_LOCAL_CRITIC=1
export DRL_MULTI_LOCAL_CRITIC_GEOMETRY_ONLY=0
export DRL_MULTI_LOCAL_CRITIC_CONTEXT_MODE=ego_motion
export DRL_MULTI_LOCAL_CRITIC_MAX_AGENTS=5
export DRL_MULTI_CRITIC_INTERACTION_FRACTION=0.0

exec "$PROJECT_ROOT/scripts/start_training_dense_simple_td3_hparam_b.sh"
