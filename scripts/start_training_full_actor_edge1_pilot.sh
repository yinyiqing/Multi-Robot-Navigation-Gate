#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIEW_DIR="$PROJECT_ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/edge1_full_horizon_v1"

export DRL_MULTI_TRAIN_MANIFEST="$VIEW_DIR/train.json.gz"
export DRL_MULTI_EVAL_MANIFEST="$VIEW_DIR/validation_monitor_50.json.gz"
export DRL_MULTI_TRAIN_FILE_NAME="${DRL_MULTI_TRAIN_FILE_NAME:-full_actor_edge1_n5_seed20260803_pilot_v1}"
export DRL_MULTI_TRAINING_VERSION="full-actor-edge1-from-5a-pilot-v1"
export DRL_MULTI_SEED="${DRL_MULTI_SEED:-20260803}"
export DRL_MULTI_MAX_EPOCHS="${DRL_MULTI_MAX_EPOCHS:-3}"
export DRL_MULTI_EVAL_FREQ_AGENT_SAMPLES="${DRL_MULTI_EVAL_FREQ_AGENT_SAMPLES:-5000}"
export DRL_MULTI_EVAL_EPISODES="${DRL_MULTI_EVAL_EPISODES:-50}"
export DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS="${DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS:-6500}"
export DRL_MULTI_EARLY_STOP_PATIENCE=0
export DRL_MULTI_ROS_PORT="${DRL_MULTI_ROS_PORT:-14221}"
export DRL_MULTI_GAZEBO_PORT="${DRL_MULTI_GAZEBO_PORT:-14321}"

exec "$PROJECT_ROOT/scripts/start_training_independent_dense_actor_from_5a.sh"
