#!/bin/bash

# Start vLLM server with Mistral Nemo 12B (40k context)
# Tier 1: Data Source Agents
# - 40k context fits comfortably in 24GB VRAM
# - Fast at long contexts
# - 66.7 MMLU (solid reasoning)

set -e

# Activate venv
source /home/antons-gs/enlitens-ai/venv/bin/activate

MODEL_PATH="/home/antons-gs/enlitens-ai/models/mistral-nemo-12b-instruct"
PORT=8000
LOG_FILE="/home/antons-gs/enlitens-ai/logs/vllm_mistral_nemo.log"

echo "🚀 Starting vLLM with Mistral Nemo 12B (40k context)..."
echo "📊 Model: $MODEL_PATH"
echo "🌐 Port: $PORT"
echo "📝 Logs: $LOG_FILE"

# Kill any existing vLLM processes
pkill -9 -f "vllm.entrypoints" || true
sleep 2

# Clear GPU memory
nvidia-smi --gpu-reset || true
sleep 2

# Allow extending context beyond model config
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Start vLLM server
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port $PORT \
    --host 0.0.0.0 \
    --tensor-parallel-size 1 \
    --max-model-len 40960 \
    --gpu-memory-utilization 0.85 \
    --dtype auto \
    --trust-remote-code \
    --disable-log-requests \
    --max-num-seqs 4 \
    --enable-prefix-caching \
    > "$LOG_FILE" 2>&1 &

echo "✅ vLLM server starting..."
echo "⏳ Waiting for server to be ready..."

# Wait for server to be ready
for i in {1..60}; do
    if curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; then
        echo "✅ Server is ready!"
        echo ""
        echo "📊 Model info:"
        curl -s http://localhost:$PORT/v1/models | python3 -m json.tool
        echo ""
        echo "🎯 Usage:"
        echo "   - Context: 128k tokens (fits in VRAM!)"
        echo "   - Output: 32k tokens max"
        echo "   - Agents: Data Source Agents (Persona, Liz Voice, St. Louis, etc.)"
        echo ""
        echo "📝 View logs: tail -f $LOG_FILE"
        exit 0
    fi
    sleep 2
done

echo "❌ Server failed to start. Check logs: $LOG_FILE"
exit 1

