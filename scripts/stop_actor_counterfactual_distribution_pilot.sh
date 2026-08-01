#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-}"
case "$PROFILE" in
  smoke|pilot) ;;
  *) echo "Usage: $0 <smoke|pilot>" >&2; exit 2 ;;
esac

PID_FILE="$PROJECT_ROOT/.g4_counterfactual_distribution_${PROFILE}.pid"
if [[ ! -f "$PID_FILE" ]]; then
  echo "No G4 counterfactual distribution $PROFILE is running."
  exit 0
fi
pid="$(tr -d '[:space:]' < "$PID_FILE")"
if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
fi
unlink "$PID_FILE"
echo "Stopped G4 counterfactual distribution $PROFILE."
