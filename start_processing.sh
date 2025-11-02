#!/bin/bash
# Start Enlitens Multi-Agent Processing
# This script ensures clean startup and proper monitoring

set -e

# Always operate from the repository root (directory containing this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs"

echo "🚀 ENLITENS MULTI-AGENT PROCESSING SYSTEM"
echo "=========================================="
echo ""

# Kill any existing processes
echo "🧹 Cleaning up old processes..."
pkill -f "process_multi_agent" 2>/dev/null || true
sleep 2

# Clean up old output files
echo "🧹 Cleaning up old output files..."
rm -f *.json *.json.temp 2>/dev/null || true
rm -f *.log 2>/dev/null || true
rm -f "$LOG_DIR"/*.log 2>/dev/null || true

# Create fresh logs directory
mkdir -p "$LOG_DIR"

# Check GPU status
echo ""
echo "🔥 GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.used,temperature.gpu --format=csv,noheader
echo ""

# Check vLLM status
echo "🤖 Checking vLLM API..."
if curl -s http://localhost:8000/v1/models > /dev/null; then
    echo "✅ vLLM server is running"
    echo "📦 Available models:"
    curl -s http://localhost:8000/v1/models | python3 -c "import sys,json; [print(f\"   - {m['id']}\") for m in json.load(sys.stdin)['data']]"
else
    echo "❌ vLLM inference service is not responding!"
    exit 1
fi

echo ""
echo "🎯 Starting multi-agent processing..."
echo "📁 Input: /home/antons-gs/enlitens-ai/enlitens_corpus/input_pdfs/"
echo "📄 Output: enlitens_knowledge_base_$(date +%Y%m%d_%H%M%S).json"
echo "📊 Log: enlitens_complete_processing.log"
echo ""

# Set timestamp for output file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="enlitens_knowledge_base_${TIMESTAMP}.json"

# Optional St. Louis report argument
ST_REPORT="/home/antons-gs/enlitens-ai/enlitens_corpus/st_louis_health_report.pdf"
ST_ARG=()
if [ -f "$ST_REPORT" ]; then
    ST_ARG=(--st-louis-report "$ST_REPORT")
else
    echo "ℹ️  St. Louis report not found at $ST_REPORT — continuing without it."
fi

# Run the processing
python3 "$SCRIPT_DIR/process_multi_agent_corpus.py" \
    --input-dir /home/antons-gs/enlitens-ai/enlitens_corpus/input_pdfs \
    --output-file "${OUTPUT_FILE}" \
    "${ST_ARG[@]}"

echo ""
echo "✅ Processing complete!"
echo "📄 Output file: ${OUTPUT_FILE}"
echo "📊 Log file: enlitens_complete_processing.log"

