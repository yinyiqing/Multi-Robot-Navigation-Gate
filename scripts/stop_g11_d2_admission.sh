#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$PROJECT_ROOT/.g11_d2_admission.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No managed G11-D2 admission is running."
  exit 0
fi
pid="$(tr -d '[:space:]' < "$PID_FILE")"
if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
  pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
  kill -- "-$pgid" 2>/dev/null || kill "$pid" 2>/dev/null || true
  echo "Stopped G11-D2 admission process group $pgid."
else
  echo "Managed G11-D2 admission is not active."
fi
unlink "$PID_FILE"
