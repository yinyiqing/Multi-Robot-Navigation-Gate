#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DRL_MULTI_TRAIN_FILE_NAME=capacity_original_width_r1_n5_seed20260810
export DRL_MULTI_EXPERIMENT_LABEL=G12-R1-original-width-control
export DRL_MULTI_PID_FILE="$PROJECT_ROOT/.g12_capacity_r1.pid"
exec "$PROJECT_ROOT/scripts/stop_training_dense_simple_td3_hparam_a.sh"
