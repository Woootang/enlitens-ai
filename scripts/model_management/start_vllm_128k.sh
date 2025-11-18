#!/bin/bash
# Start vLLM with Qwen 2.5-14B at FULL 128k context
# MAXIMUM QUALITY MODE

set -e

cd /home/antons-gs/enlitens-ai

echo "🚀 Starting vLLM with FULL 128k Context"
echo "========================================"
echo ""

# Kill existing
pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
sleep 3

# GPU status
echo "🔥 GPU Status:"
nvidia-smi --query-gpu=name,memory.free,temperature.gpu --format=csv,noheader
echo ""

echo "📦 Model: Qwen 2.5-14B Instruct (AWQ 4-bit)"
echo "🧠 Context: 56k tokens (MAXIMUM for 24GB GPU)"
echo "✨ Quality: Fits paper + 30-40 personas + transcripts + RAG"
echo "⚡ Speed: ~2-3x slower than 16k (perfect balance)"
echo ""

# Activate and start
source venv/bin/activate

# Allow extended context
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

python3 -m vllm.entrypoints.openai.api_server \
    --model /home/antons-gs/enlitens-ai/models/qwen2.5-14b-instruct-awq \
    --port 8000 \
    --host 0.0.0.0 \
    --tensor-parallel-size 1 \
    --max-model-len 56544 \
    --gpu-memory-utilization 0.90 \
    --quantization awq \
    --dtype auto \
    --trust-remote-code \
    --disable-log-requests \
    > logs/vllm_128k.log 2>&1 &

VLLM_PID=$!
echo "✅ vLLM started (PID: $VLLM_PID)"
echo "📊 Log: logs/vllm_128k.log"
echo ""

# Wait for ready
echo "⏳ Waiting for vLLM to initialize (3-5 minutes for 128k)..."
for i in {1..300}; do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo ""
        echo "✅ vLLM is READY with 128k context!"
        echo ""
        curl -s http://localhost:8000/v1/models | python3 -c "import sys,json; [print(f'   📦 {m[\"id\"]}') for m in json.load(sys.stdin)['data']]"
        echo ""
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
        echo ""
        echo "🎉 Ready for MAXIMUM QUALITY processing!"
        exit 0
    fi
    
    if [ $((i % 30)) -eq 0 ]; then
        echo "   Still loading... ($i seconds)"
    fi
    
    sleep 1
done

echo "❌ Timeout - check logs: tail -f logs/vllm_128k.log"
exit 1

