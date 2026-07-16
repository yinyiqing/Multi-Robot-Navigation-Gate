#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec "$PROJECT_ROOT/scripts/stop_test_detached_multi_stage2_to_5d_geo_critic_from_5a_guarded_best.sh"
