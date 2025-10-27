#!/bin/bash
# Start Enlitens Multi-Agent Processing
# This script ensures clean startup and proper monitoring

set -e

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
rm -f logs/*.log 2>/dev/null || true

# Create fresh logs directory
mkdir -p logs

# Check GPU status
echo ""
echo "🔥 GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.used,temperature.gpu --format=csv,noheader
echo ""

# Check Ollama status
echo "🤖 Checking Ollama..."
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✅ Ollama is running"
    echo "📦 Available models:"
    curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; [print(f\"   - {m['name']}\") for m in json.load(sys.stdin)['models']]"
else
    echo "❌ Ollama is not responding!"
    exit 1
fi

echo ""
echo "🎯 Starting multi-agent processing..."
echo "📁 Input: enlitens_corpus/input_pdfs/"
echo "📄 Output: enlitens_knowledge_base_$(date +%Y%m%d_%H%M%S).json"
echo "📊 Log: enlitens_complete_processing.log"
echo ""

# Set timestamp for output file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="enlitens_knowledge_base_${TIMESTAMP}.json"

# Run the processing
python3 process_multi_agent_corpus.py \
    --input-dir enlitens_corpus/input_pdfs \
    --output-file "${OUTPUT_FILE}" \
    --st-louis-report st_louis_health_report.pdf

echo ""
echo "✅ Processing complete!"
echo "📄 Output file: ${OUTPUT_FILE}"
echo "📊 Log file: enlitens_complete_processing.log"

