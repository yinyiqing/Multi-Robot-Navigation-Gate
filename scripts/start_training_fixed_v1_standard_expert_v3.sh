#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DRL_MULTI_TRAIN_FILE_NAME="${DRL_MULTI_TRAIN_FILE_NAME:-standard_expert_5d_fixed_v1_timeoutfix_v3}"
export DRL_MULTI_ACTOR_ANCHOR_WEIGHT="${DRL_MULTI_ACTOR_ANCHOR_WEIGHT:-1.0}"
export DRL_MULTI_LOCAL_CRITIC_ACTOR_UPDATE_DELAY_STEPS="${DRL_MULTI_LOCAL_CRITIC_ACTOR_UPDATE_DELAY_STEPS:-5000}"
export DRL_MULTI_TRAINING_VERSION="standard-expert-fixed-v1-timeoutfix-v3"
export DRL_MULTI_MAX_EPOCHS="${DRL_MULTI_MAX_EPOCHS:-3}"
export DRL_MULTI_EVAL_EPISODES="${DRL_MULTI_EVAL_EPISODES:-100}"

exec "$SCRIPT_DIR/start_training_fixed_v1_standard_expert_v2.sh"
