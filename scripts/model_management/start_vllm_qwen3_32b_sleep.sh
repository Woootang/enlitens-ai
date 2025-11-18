#!/bin/bash
# Start vLLM server for Qwen3-32B with Sleep Mode enabled
# Port: 8001
# Context: 40k tokens (realistic for 24GB VRAM with larger model)
# Sleep Mode: Allows hibernation without killing process

set -e

# Configuration
MODEL_PATH="/home/antons-gs/enlitens-ai/models/qwen3-32b-instruct-awq"
PORT=8001
LOG_DIR="/home/antons-gs/enlitens-ai/logs"
LOG_FILE="$LOG_DIR/vllm_qwen3_32b_sleep.log"

# Create log directory
mkdir -p "$LOG_DIR"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model not found: $MODEL_PATH"
    exit 1
fi

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Port $PORT already in use, skipping startup"
    exit 0
fi

echo "🚀 Starting Qwen3-32B with Sleep Mode on port $PORT..."

# Activate virtual environment
source /home/antons-gs/enlitens-ai/venv/bin/activate

# Enable Sleep Mode and dev endpoints
export VLLM_SERVER_DEV_MODE=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Start vLLM server with Sleep Mode
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port $PORT \
    --host 0.0.0.0 \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --quantization awq \
    --dtype auto \
    --swap-space 8 \
    --max-num-seqs 1 \
    --enable-prefix-caching \
    --enable-sleep-mode \
    --trust-remote-code \
    --disable-log-requests \
    > "$LOG_FILE" 2>&1 &

echo "✅ Qwen3-32B starting on port $PORT (Sleep Mode enabled)"
echo "📝 Logs: $LOG_FILE"

