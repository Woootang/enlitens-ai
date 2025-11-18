#!/usr/bin/env bash
#
# Reset dashboard-visible state (logs + status files) so the next pipeline run
# starts from a clean slate.  Safe to run repeatedly.
#
set -euo pipefail

ROOT="/home/antons-gs/enlitens-ai"
LOG_DIR="$ROOT/logs"
LOCAL_STATUS="$LOG_DIR/local_status.json"
PROCESSING_LOG="$LOG_DIR/processing.log"
LEGACY_LOG="$LOG_DIR/enlitens_complete_processing.log"

echo "🧹 Resetting dashboard state..."

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Truncate primary processing log files
if [[ -f "$PROCESSING_LOG" ]]; then
  : > "$PROCESSING_LOG"
  echo "  • Cleared $PROCESSING_LOG"
else
  touch "$PROCESSING_LOG"
  echo "  • Created $PROCESSING_LOG"
fi

if [[ -f "$LEGACY_LOG" ]]; then
  : > "$LEGACY_LOG"
  echo "  • Cleared $LEGACY_LOG"
fi

# Remove snapshot status so UI knows there is no active run yet
if [[ -f "$LOCAL_STATUS" ]]; then
  rm -f "$LOCAL_STATUS"
  echo "  • Removed $LOCAL_STATUS"
fi

# Clear cached dashboard previews
if [[ -d "$LOG_DIR/local_models" ]]; then
  find "$LOG_DIR/local_models" -type f -maxdepth 1 -name "*.log" -exec rm -f {} \;
  echo "  • Purged local model log cache"
fi

echo "✅ Dashboard state reset. Start your pipeline and the UI will stream fresh data."

