#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DRL_MULTI_TRAIN_FILE_NAME="${DRL_MULTI_TRAIN_FILE_NAME:-full_actor_edge1_n5_seed20260803_pilot_v1}"
exec "$PROJECT_ROOT/scripts/stop_training_independent_dense_actor_from_5a.sh"
