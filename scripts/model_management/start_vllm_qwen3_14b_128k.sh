#!/bin/bash

# Start vLLM server with Qwen3-14B (128k context)
# Tier 2: Research Agents
# - 128k context with minimal CPU offload
# - 85 MMLU (near 70B-level reasoning)
# - Best balance of quality and speed

set -e

# Activate venv
source /home/antons-gs/enlitens-ai/venv/bin/activate

MODEL_PATH="/home/antons-gs/enlitens-ai/models/qwen3-14b-instruct-awq"
PORT=8000
LOG_FILE="/home/antons-gs/enlitens-ai/logs/vllm_qwen3_14b.log"

GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.98}
MIN_FREE_MEM_MB=${MIN_FREE_MEM_MB:-20000} # require ~20GB free on 3090

echo "🚀 Starting vLLM with Qwen3-14B (target 128k context)..."
echo "📊 Model: $MODEL_PATH"
echo "🌐 Port: $PORT"
echo "📝 Logs: $LOG_FILE"
echo "🧠 GPU memory target: ${GPU_MEMORY_UTIL}"

# Ensure GPU is free enough before launching
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "🔍 Checking current GPU processes..."
    nvidia-smi
    FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1)
    if [ "${FREE_MEM}" -lt "${MIN_FREE_MEM_MB}" ]; then
        echo "❌ Only ${FREE_MEM} MiB free on GPU. Need at least ${MIN_FREE_MEM_MB} MiB."
        echo "   Kill leftover processes with: sudo fuser -v /dev/nvidia*"
        echo "   or: nvidia-smi --query-compute-apps=pid,process_name --format=csv"
        exit 1
    fi
else
    echo "⚠️ nvidia-smi not found; skipping GPU availability check."
fi

# Kill any existing vLLM processes
pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
sleep 2

# Allow extending context beyond model config (for RoPE scaling)
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Start vLLM server with CPU offload
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port $PORT \
    --host 0.0.0.0 \
    --tensor-parallel-size 1 \
    --max-model-len 65536 \
    --gpu-memory-utilization "${GPU_MEMORY_UTIL}" \
    --quantization awq \
    --dtype auto \
    --enforce-eager \
    --trust-remote-code \
    --max-num-seqs 1 \
    --enable-prefix-caching \
    --swap-space 32 \
    --cpu-offload-gb 40 \
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
        echo "   - Context: 128k tokens (20GB CPU offload)"
        echo "   - Output: 32k tokens max"
        echo "   - Agents: Research Agents (Science, Clinical, etc.)"
        echo ""
        echo "📝 View logs: tail -f $LOG_FILE"
        exit 0
    fi
    sleep 2
done

echo "❌ Server failed to start. Check logs: $LOG_FILE"
exit 1

