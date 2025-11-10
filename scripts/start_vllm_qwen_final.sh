#!/bin/bash
# Start vLLM with Qwen 2.5-14B Instruct (AWQ)
# Final working version for PDF processing

set -e

echo "🚀 Starting vLLM with Qwen 2.5-14B Instruct"
echo "============================================"
echo ""

# Kill any existing vLLM processes
echo "🧹 Cleaning up old vLLM processes..."
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 2

# Check GPU
echo "🔥 GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free,temperature.gpu --format=csv,noheader
echo ""

# Model path
MODEL_PATH="/home/antons-gs/enlitens-ai/models/qwen2.5-14b-instruct-awq"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model not found at: $MODEL_PATH"
    exit 1
fi

echo "📦 Model: Qwen 2.5-14B Instruct (AWQ)"
echo "📁 Path: $MODEL_PATH"
echo "🔧 Quantization: AWQ 4-bit (~8GB VRAM)"
echo "🧠 Context: FULL 128k tokens - MAXIMUM QUALITY MODE"
echo "✨ Quality: Fits PDF + ALL personas + transcripts + RAG + external search"
echo "⚡ Speed: ~4-5x slower per doc, but MUCH better outputs"
echo ""

echo "🎯 Starting vLLM server..."
echo "   Port: 8000"
echo "   Max model length: 131072 (128k - FULL CONTEXT!)"
echo "   GPU memory utilization: 0.90"
echo ""

# Activate venv
source /home/antons-gs/enlitens-ai/venv/bin/activate

# Start vLLM in background
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port 8000 \
    --host 0.0.0.0 \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.90 \
    --quantization awq \
    --dtype auto \
    --trust-remote-code \
    --disable-log-requests \
    > logs/vllm_qwen.log 2>&1 &

VLLM_PID=$!
echo "✅ vLLM started (PID: $VLLM_PID)"
echo "📊 Log: logs/vllm_qwen.log"
echo ""

# Wait for server to be ready
echo "⏳ Waiting for vLLM server to initialize (this takes 2-3 minutes)..."
for i in {1..180}; do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "✅ vLLM server is ready!"
        echo ""
        
        # Show loaded model
        echo "📦 Loaded models:"
        curl -s http://localhost:8000/v1/models | python3 -c "import sys,json; [print(f\"   - {m['id']}\") for m in json.load(sys.stdin)['data']]"
        echo ""
        
        echo "🎉 Qwen 2.5-14B is ready for PDF processing!"
        echo ""
        echo "✨ Quality upgrade: 2x better than Mistral-7B"
        echo "   - 14B parameters (vs 7B)"
        echo "   - 128k context (vs 32k)"
        echo "   - Better reasoning & extraction"
        echo ""
        echo "To monitor:"
        echo "   tail -f logs/vllm_qwen.log"
        echo ""
        echo "To stop:"
        echo "   pkill -f vllm.entrypoints"
        
        exit 0
    fi
    
    if [ $((i % 30)) -eq 0 ]; then
        echo "   Still loading... ($i seconds)"
    fi
    
    sleep 1
done

echo "❌ vLLM server failed to start within 3 minutes"
echo "Check logs: tail -f logs/vllm_qwen.log"
exit 1

