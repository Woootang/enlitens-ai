#!/bin/bash
# Start vLLM with Qwen3 14B at 60k context using CPU offloading
# Uses GPU (24GB) + CPU RAM (64GB) for maximum context

set -e

echo "🚀 Starting vLLM with 60k Context (GPU + CPU Hybrid)"
echo "====================================================="
echo ""

# Kill existing
pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
sleep 3

# GPU status
echo "🔥 GPU Status:"
nvidia-smi --query-gpu=name,memory.free,temperature.gpu --format=csv,noheader
echo ""

MODEL_PATH="/home/antons-gs/enlitens-ai/models/qwen3-14b-instruct-awq"

if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model not found at: $MODEL_PATH"
    exit 1
fi

echo "📦 Model: Qwen3 14B (AWQ 4-bit)"
echo "🧠 Context: 60k tokens (GPU + CPU hybrid)"
echo "💾 VRAM: 24GB GPU + 40GB CPU RAM"
echo "✨ Quality: Fits paper + ALL personas + transcripts + RAG + external search"
echo "⚡ Speed: ~3-4x slower than 40k (worth it for quality)"
echo ""

# Activate venv
source /home/antons-gs/enlitens-ai/venv/bin/activate

# Allow extended context
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Start with CPU offloading
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port 8000 \
    --host 0.0.0.0 \
    --tensor-parallel-size 1 \
    --max-model-len 61440 \
    --gpu-memory-utilization 0.85 \
    --swap-space 40 \
    --quantization awq \
    --dtype auto \
    --trust-remote-code \
    --disable-log-requests \
    > logs/vllm_qwen_60k.log 2>&1 &

VLLM_PID=$!
echo "✅ vLLM started (PID: $VLLM_PID)"
echo "📊 Log: logs/vllm_qwen_60k.log"
echo ""

# Wait for ready
echo "⏳ Waiting for vLLM to initialize (3-5 minutes)..."
for i in {1..300}; do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo ""
        echo "✅ vLLM is READY with 60k context!"
        echo ""
        curl -s http://localhost:8000/v1/models | python3 -c "import sys,json; [print(f'   📦 {m[\"id\"]}') for m in json.load(sys.stdin)['data']]"
        echo ""
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
        echo ""
        echo "🎉 Ready for HIGH QUALITY processing!"
        exit 0
    fi
    
    if [ $((i % 30)) -eq 0 ]; then
        echo "   Still loading... ($i seconds)"
    fi
    
    sleep 1
done

echo "❌ Timeout - check logs: tail -f logs/vllm_qwen_60k.log"
exit 1

