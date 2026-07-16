#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export DRL_MULTI_TEST_ACTOR_MODE=residual
export DRL_MULTI_RESIDUAL_HIDDEN_DIM="${DRL_MULTI_RESIDUAL_HIDDEN_DIM:-128}"
export DRL_MULTI_RESIDUAL_SCALE="${DRL_MULTI_RESIDUAL_SCALE:-0.15}"
export DRL_MULTI_TEST_LOG_TAG="${DRL_MULTI_TEST_LOG_TAG:-RESIDUAL_5D}"

exec "$PROJECT_ROOT/scripts/start_test_detached_multi_curriculum.sh" \
  stage4_asym_dense_5_moderate \
  "${DRL_MULTI_TEST_FILE_NAME:-TD3_multi_dense5_moderate_residual_from_5d_best}"
