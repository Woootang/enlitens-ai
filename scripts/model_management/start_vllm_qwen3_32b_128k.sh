#!/bin/bash

# Start vLLM server with Qwen3-32B (40k context)
# Tier 3: Final Writer Agents
# - 40k context fits with this larger model
# - ~90 MMLU (GPT-4 level)
# - Maximum quality for client-facing content

set -e

# Activate venv
source /home/antons-gs/enlitens-ai/venv/bin/activate

MODEL_PATH="/home/antons-gs/enlitens-ai/models/qwen3-32b-instruct-awq"
PORT=8000
LOG_FILE="/home/antons-gs/enlitens-ai/logs/vllm_qwen3_32b.log"

echo "🚀 Starting vLLM with Qwen3-32B (40k context)..."
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
    --quantization awq \
    --dtype auto \
    --trust-remote-code \
    --disable-log-requests \
    --max-num-seqs 2 \
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
        echo "   - Context: 128k tokens (40GB CPU offload)"
        echo "   - Output: 32k tokens max"
        echo "   - Agents: Final Writer Agents (Blog, Social, etc.)"
        echo ""
        echo "📝 View logs: tail -f $LOG_FILE"
        exit 0
    fi
    sleep 2
done

echo "❌ Server failed to start. Check logs: $LOG_FILE"
exit 1

