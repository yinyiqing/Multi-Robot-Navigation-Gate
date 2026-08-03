#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export DRL_MULTI_EXPERIMENT_LABEL=edge1-full-actor-from-5a
export DRL_MULTI_TRAIN_FILE_NAME="${DRL_MULTI_TRAIN_FILE_NAME:-full_actor_edge1_from_5a_s20260803}"

exec "$PROJECT_ROOT/scripts/stop_training_dense_simple_td3_hparam_a.sh"
