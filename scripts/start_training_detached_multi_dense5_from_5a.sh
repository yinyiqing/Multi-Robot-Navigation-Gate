#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export DRL_MULTI_TRAIN_FILE_NAME="${DRL_MULTI_TRAIN_FILE_NAME:-TD3_multi_dense5_from_5a_geo}"
export DRL_MULTI_LOAD_MODEL_NAME="${DRL_MULTI_LOAD_MODEL_NAME:-TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best}"
export DRL_MULTI_LOAD_ACTOR_ONLY="${DRL_MULTI_LOAD_ACTOR_ONLY:-1}"
export DRL_MULTI_RESUME_TRAINING="${DRL_MULTI_RESUME_TRAINING:-0}"
export DRL_MULTI_TRAINING_VERSION="${DRL_MULTI_TRAINING_VERSION:-dense5-from-5a-geo-v1}"
export DRL_MULTI_ROS_PORT="${DRL_MULTI_ROS_PORT:-13631}"
export DRL_MULTI_GAZEBO_PORT="${DRL_MULTI_GAZEBO_PORT:-13731}"

exec "$PROJECT_ROOT/scripts/start_training_detached_multi_curriculum.sh" stage4_asym_dense_5
