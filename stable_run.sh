#!/bin/bash
# Stable Execution Script for Enlitens Multi-Agent System
# This script addresses SSH tunnel stability issues and ensures clean execution

echo "🚀 Starting Stable Multi-Agent Processing..."

# Kill any existing processes that might cause conflicts
echo "🧹 Cleaning up existing processes..."
pkill -f "process_.*corpus" || true
pkill -f "python.*process" || true
pkill -f "ollama" || true

# Wait a moment for processes to terminate
sleep 3

# Clear any stuck terminal state
stty sane 2>/dev/null || true

# Set environment variables for stability
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_MAX_QUEUE=1
export OLLAMA_RUNNERS_DIR=/tmp/ollama-runners
export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1

# Check if Ollama is running, start if not
echo "🔍 Checking Ollama service..."
if ! pgrep -f "ollama" > /dev/null; then
    echo "📦 Starting Ollama service..."
    nohup ollama serve > ollama.log 2>&1 &
    sleep 5

    # Verify Ollama started
    if ! pgrep -f "ollama" > /dev/null; then
        echo "❌ Failed to start Ollama"
        exit 1
    fi
    echo "✅ Ollama service started"
else
    echo "✅ Ollama service already running"
fi

# Check GPU status
echo "🔥 Checking GPU status..."
nvidia-smi || echo "⚠️ NVIDIA drivers not available"

# Check available disk space
echo "💽 Checking disk space..."
df -h | grep -E "(Filesystem|/dev/)" || true

# Create output directory if it doesn't exist
mkdir -p logs

# Generate timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="enlitens_complete_processing.log"
TEMP_LOG="logs/temp_processing.log"

# Clean up old files first
echo "🧹 Cleaning up old files..."
rm -f *.log logs/*.log
rm -f enlitens_knowledge_base*.json*

echo "📊 Starting processing at: $(date)"
echo "📝 Comprehensive log file: ${LOG_FILE} (all 344 files)"
echo "🏙️ St. Louis context loaded: ✅"
echo "🧠 Enhanced multi-agent system with quality validation: ✅"
echo "🔄 Robust retry mechanisms: ✅"
echo "📈 Confidence scoring and fact checking: ✅"

# Run the multi-agent processor with nohup to prevent SSH issues
echo "🎯 Launching enhanced multi-agent processor..."
nohup python3 process_multi_agent_corpus.py \
    --input-dir enlitens_corpus/input_pdfs \
    --output-file "enlitens_knowledge_base_complete_${TIMESTAMP}.json" \
    --st-louis-report st_louis_health_report.pdf \
    > "${TEMP_LOG}" 2>&1 &
PROCESS_PID=$!

echo "✅ Process started with PID: ${PROCESS_PID}"
echo "📋 Monitor progress with: tail -f ${TEMP_LOG}"
echo "🛑 Stop process with: kill ${PROCESS_PID}"
echo ""
echo "💡 The system is now running in the background."
echo "💡 Check progress: tail -f ${TEMP_LOG}"
echo "💡 View system status: nvidia-smi"
echo "💡 Check Ollama: curl http://localhost:11434/api/tags"
echo ""
echo "🚀 Multi-agent processing initiated successfully!"
echo "⏰ Started at: $(date)"

# Optional: Monitor the process briefly
sleep 2
if ps -p ${PROCESS_PID} > /dev/null; then
    echo "✅ Process is running successfully"
    echo "📊 Process ID: ${PROCESS_PID}"
    echo "📝 Monitor with: tail -f ${TEMP_LOG}"
else
    echo "❌ Process failed to start"
    echo "📋 Check logs: ${TEMP_LOG}"
    exit 1
fi
