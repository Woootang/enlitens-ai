#!/usr/bin/env bash
# Usage: ./scripts/stop_local_model.sh medgemma
set -euo pipefail
MODEL_KEY="${1:?model key required (medgemma|llama)}"
PID_FILE="logs/${MODEL_KEY}.pid"
if [[ ! -f "$PID_FILE" ]]; then
  echo "No PID file found for $MODEL_KEY ($PID_FILE). Nothing to stop."
  exit 0
fi
PID=$(cat "$PID_FILE")
if ps -p "$PID" > /dev/null; then
  echo "Stopping $MODEL_KEY (PID=$PID)..."
  kill "$PID"
  wait "$PID" 2>/dev/null || true
else
  echo "Process $PID not running."
fi
rm -f "$PID_FILE"
echo "GPU reset request..."
command -v nvidia-smi >/dev/null && nvidia-smi --gpu-reset || true
echo "Stop script completed."

