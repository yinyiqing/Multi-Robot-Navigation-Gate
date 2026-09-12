#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
G27="$ROOT/experiments/03_保留专门化/02_论文主线/27_反事实Reward增强Gate监督"
AUDIT="$G27/local_data/counterfactual/audit/summary.json"

[[ -f "$AUDIT" ]] || { echo "G27 P1 audit is missing" >&2; exit 2; }

set +u
source "$ROOT/env.python.sh"
set -u

python3 -u "$ROOT/scripts/train_g27_reward_aware_gate.py" "$@"
