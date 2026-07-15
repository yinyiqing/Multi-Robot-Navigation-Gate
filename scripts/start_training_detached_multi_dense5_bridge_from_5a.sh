#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export DRL_MULTI_TRAIN_FILE_NAME="${DRL_MULTI_TRAIN_FILE_NAME:-TD3_multi_dense5_bridge_from_5a}"
export DRL_MULTI_LOAD_MODEL_NAME="${DRL_MULTI_LOAD_MODEL_NAME:-TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best}"
export DRL_MULTI_LOAD_ACTOR_ONLY="${DRL_MULTI_LOAD_ACTOR_ONLY:-1}"
export DRL_MULTI_RESUME_TRAINING="${DRL_MULTI_RESUME_TRAINING:-0}"
export DRL_MULTI_ACTOR_LR="${DRL_MULTI_ACTOR_LR:-0.0000003}"
export DRL_MULTI_CRITIC_LR="${DRL_MULTI_CRITIC_LR:-0.00004}"
export DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS="${DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS:-15000}"
export DRL_MULTI_ACTOR_ANCHOR_WEIGHT="${DRL_MULTI_ACTOR_ANCHOR_WEIGHT:-0.02}"
export DRL_MULTI_TRAINING_VERSION="${DRL_MULTI_TRAINING_VERSION:-dense5-bridge-from-5a-conservative-v1}"

exec "$PROJECT_ROOT/scripts/start_training_detached_multi_curriculum.sh" stage4_asym_dense_5_bridge
