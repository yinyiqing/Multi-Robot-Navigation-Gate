#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export DRL_MULTI_EXPERIMENT_LABEL=D2b
export DRL_MULTI_TRAIN_FILE_NAME="${DRL_MULTI_TRAIN_FILE_NAME:-independent_dense_actor_simple_td3_hparam_d2b_s20260801}"

exec "$PROJECT_ROOT/scripts/stop_training_dense_simple_td3_hparam_a.sh"
