#!/usr/bin/env bash
set -euo pipefail

# I-E2-F4: isolated E2-compatible recovery Actor pilot.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROUTE="$ROOT/experiments/03_保留专门化/02_论文主线/15_E2恢复Actor诊断与训练"

export IE2_MODEL_NAME="interaction_recovery_from_e2_fourphase_s20260813"
export IE2_RUN_DIR="$ROUTE/local_data/ie2_f4_fourphase_pilot"
export IE2_LOG_DIR="$ROOT/logs/active/e2-ie2-f4-fourphase-pilot"
export IE2_TRAIN_MANIFEST="$ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/ie2_multi_conflict_v1/train.json.gz"
export IE2_EVAL_MANIFEST="$ROOT/experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/ie2_multi_conflict_v1/validation.json.gz"
export IE2_TRAINING_VERSION="e2-interaction-recovery-fourphase-pilot-v1"
export IE2_REWARD_MODE="average_plus_interaction"
export IE2_INTERACTION_STAGNATION_PENALTY="0.05"
export IE2_USE_SAFE_RECOVERY_REWARD="1"
export IE2_SAFE_RECOVERY_PENALTY="0.2"
export IE2_SAFE_RECOVERY_PROGRESS_BONUS_WEIGHT="0.8"
export IE2_SAFE_RECOVERY_IDLE_PENALTY_WEIGHT="1.0"
export IE2_SAFE_RECOVERY_ROBOT_DISTANCE="0.0"

# Let every interaction-window frame update the Actor. The rejected pilots
# filtered gradients to approaching-risk frames, excluding stalled recovery.
export IE2_ACTOR_SAFETY_FOCUSED="0"
export IE2_ACTOR_SAFETY_DISTANCE="2.0"
export IE2_ACTOR_SAFETY_MIN_CLOSING_SPEED="-1.0"
export IE2_ACTOR_UPDATE_DELAY_STEPS="21000"
export IE2_TRAIN_EVAL_EPISODES="200"

exec bash "$ROOT/scripts/run_e2_ie2_overnight_pipeline_worker.sh"
