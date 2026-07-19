#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
PID_FILE="$PROJECT_ROOT/.validation_fixed_v1_standard_v3.pid"
BASE_ACTOR="TD3_velodyne_multi_v4_curriculum_stage2_to_5d_geo_critic_from_5a_guarded_best"
V3_ACTOR="standard_expert_5d_fixed_v1_timeoutfix_v3_best"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Validation comparison is already running with PID $old_pid"
    exit 1
  fi
fi

mkdir -p "$LOG_DIR"
timestamp="$(date +%Y%m%d_%H%M%S)"
base_log="$LOG_DIR/validation500_standard_5d_${timestamp}.log"
v3_log="$LOG_DIR/validation500_standard_v3_epoch002_${timestamp}.log"
runner="$PROJECT_ROOT/scripts/run_validation_fixed_v1_standard_actor.sh"

setsid bash -lc "
  set -eo pipefail
  cleanup() {
    pgid=\"\$(ps -o pgid= -p \$\$ | tr -d ' ')\"
    ps -eo pid=,pgid= | awk -v pgid=\"\$pgid\" -v self=\"\$\$\" \
      '\$2 == pgid && \$1 != self { print \$1 }' | xargs -r kill 2>/dev/null || true
    rm -f '$PID_FILE'
  }
  trap cleanup EXIT
  '$runner' baseline_5d '$BASE_ACTOR' 12001 12101 '$base_log'
  sleep 3
  '$runner' v3_epoch002 '$V3_ACTOR' 12002 12102 '$v3_log'
" >"$LOG_DIR/validation500_standard_compare_${timestamp}.log" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "Started sequential 500-scenario standard validation comparison."
echo "PID: $(cat "$PID_FILE")"
echo "Baseline log: $base_log"
echo "V3 log: $v3_log"
