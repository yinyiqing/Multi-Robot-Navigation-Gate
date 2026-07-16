#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec "$PROJECT_ROOT/scripts/stop_training_detached_multi_curriculum.sh" stage4_asym_dense_5_moderate
