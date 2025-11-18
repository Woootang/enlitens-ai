#!/usr/bin/env bash
#
# Enlitens AI – Full Corpus Processing Launcher
# ---------------------------------------------
# This helper script bootstraps the end‑to‑end multi‑agent processing run against
# the complete 345 PDF corpus.  It wraps environment activation, sanity checks,
# log routing, and launches `process_multi_agent_corpus.py` with the correct
# absolute paths expected inside the project.
#
# Usage:
#   ./scripts/run_full_corpus.sh                # default configuration
#   ENLITENS_MONITOR_URL=http://localhost:8765/api/log ./scripts/run_full_corpus.sh
#   ./scripts/run_full_corpus.sh /path/to/pdfs  # override input directory
#
# Optional environment toggles:
#   ENLITENS_DISABLE_VECTOR_INGESTION=1  # skip vector DB ingestion when VRAM tight
#   ENLITENS_MONITOR_URL=...             # override dashboard ingestion endpoint
#   ENLITENS_OUTPUT_BASENAME=...         # customise output filename (without path)
#
set -euo pipefail

PROJECT_ROOT="/home/antons-gs/enlitens-ai"
VENV_PATH="$PROJECT_ROOT/venv/bin/activate"
INPUT_DIR_DEFAULT="$PROJECT_ROOT/enlitens_corpus/input_pdfs"
OUTPUT_DIR="$PROJECT_ROOT/enlitens_knowledge_base"
OUTPUT_BASENAME="${ENLITENS_OUTPUT_BASENAME:-enlitens_knowledge_base.json}"
OUTPUT_FILE="$OUTPUT_DIR/$OUTPUT_BASENAME"
STL_REPORT="$PROJECT_ROOT/st_louis_health_report.pdf"
LOG_DIR="$PROJECT_ROOT/logs"

# Allow optional positional override for input directory
INPUT_DIR="${1:-$INPUT_DIR_DEFAULT}"

echo "🧠 Enlitens Full-Corpus Processor"
echo "────────────────────────────────────"
echo "📁 Input directory : $INPUT_DIR"
echo "💾 Output file     : $OUTPUT_FILE"
echo "🗂  Logs directory  : $LOG_DIR"
echo "📊 Monitor endpoint: ${ENLITENS_MONITOR_URL:-http://localhost:8765/api/log}"
echo ""

# Sanity checks
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "❌ Input directory not found: $INPUT_DIR" >&2
    exit 1
fi

if [[ ! -f "$STL_REPORT" ]]; then
    echo "⚠️  St. Louis health report missing at $STL_REPORT (continuing without additional context)"
    STL_REPORT=""
fi

if [[ ! -f "$VENV_PATH" ]]; then
    echo "❌ Python virtual environment not found at $VENV_PATH" >&2
    exit 1
fi

# Prepare directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Activate environment
source "$VENV_PATH"

# Export monitor endpoint default if user did not supply one
export ENLITENS_MONITOR_URL="${ENLITENS_MONITOR_URL:-http://localhost:8765/api/log}"

# Fix CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Stamp run metadata
RUN_TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"
echo "🚀 Launching multi-agent processing @ $RUN_TIMESTAMP"
echo "   Vector ingestion: ${ENLITENS_DISABLE_VECTOR_INGESTION:-enabled}"
echo ""

CMD=(python "$PROJECT_ROOT/process_multi_agent_corpus.py"
     --input-dir "$INPUT_DIR"
     --output-file "$OUTPUT_FILE")

if [[ -n "$STL_REPORT" ]]; then
    CMD+=(--st-louis-report "$STL_REPORT")
fi

echo "🔧 Command:"
printf '   %q ' "${CMD[@]}"
echo ""
echo ""

"${CMD[@]}"

EXIT_CODE=$?
echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "✅ Run completed successfully."
    echo "   Knowledge base: $OUTPUT_FILE"
else
    echo "❌ Run exited with code $EXIT_CODE."
fi

exit $EXIT_CODE

