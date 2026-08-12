#!/usr/bin/env bash
set -euo pipefail

# Fresh branch: same protocol as the old local-critic actor training,
# but warm-started from the newly retrained ordinary Actor instead of 5A.
export DRL_MULTI_LOAD_MODEL_NAME="${DRL_MULTI_LOAD_MODEL_NAME:-current_generalist_n5_efficiency_e2_s20260810_best}"
export DRL_MULTI_TRAIN_FILE_NAME="${DRL_MULTI_TRAIN_FILE_NAME:-current_generalist_from_e2_local_critic_s20260811}"
export DRL_MULTI_TRAINING_VERSION="${DRL_MULTI_TRAINING_VERSION:-current-generalist-from-e2-local-critic-v1}"
export DRL_MULTI_EXPERIMENT_LABEL="${DRL_MULTI_EXPERIMENT_LABEL:-current-generalist-from-e2-local-critic}"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start_training_current_generalist_from_5a_local_critic.sh"
