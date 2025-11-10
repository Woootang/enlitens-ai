#!/bin/bash
# Start vLLM with Qwen 2.5-14B Instruct (8-bit GPTQ)
# Higher quality than 4-bit AWQ

set -e

echo "🚀 Starting vLLM with Qwen 2.5-14B Instruct (8-bit)"
echo "===================================================="
echo ""

# Kill any existing vLLM processes
echo "🧹 Cleaning up old vLLM processes..."
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 3

# Check GPU
echo "🔥 GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free,temperature.gpu --format=csv,noheader
echo ""

# Model path
MODEL_PATH="/home/antons-gs/enlitens-ai/models/qwen2.5-14b-instruct-gptq-int8"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model not found at: $MODEL_PATH"
    echo "   Run: huggingface-cli download Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8 --local-dir $MODEL_PATH"
    exit 1
fi

echo "📦 Model: Qwen 2.5-14B Instruct (8-bit GPTQ)"
echo "📁 Path: $MODEL_PATH"
echo "🔧 Quantization: GPTQ 8-bit (~14GB VRAM)"
echo "🧠 Context: 128k tokens (using 16k for speed)"
echo "✨ Quality: +5% better than 4-bit AWQ"
echo ""

echo "🎯 Starting vLLM server..."
echo "   Port: 8000"
echo "   Max model length: 12288 (adjusted for 8-bit memory)"
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
    --max-model-len 12288 \
    --gpu-memory-utilization 0.90 \
    --quantization gptq \
    --dtype auto \
    --trust-remote-code \
    --disable-log-requests \
    > logs/vllm_qwen_8bit.log 2>&1 &

VLLM_PID=$!
echo "✅ vLLM started (PID: $VLLM_PID)"
echo "📊 Log: logs/vllm_qwen_8bit.log"
echo ""

# Wait for server to be ready
echo "⏳ Waiting for vLLM server to initialize (this takes 3-5 minutes for 8-bit)..."
for i in {1..300}; do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "✅ vLLM server is ready!"
        echo ""
        
        # Show loaded model
        echo "📦 Loaded models:"
        curl -s http://localhost:8000/v1/models | python3 -c "import sys,json; [print(f\"   - {m['id']}\") for m in json.load(sys.stdin)['data']]"
        echo ""
        
        echo "🎉 Qwen 2.5-14B (8-bit) is ready for PDF processing!"
        echo ""
        echo "✨ Quality upgrade from 4-bit:"
        echo "   - 8-bit quantization (vs 4-bit)"
        echo "   - ~5% better accuracy"
        echo "   - More precise reasoning"
        echo "   - Better entity extraction"
        echo ""
        echo "📊 Memory usage:"
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
        echo ""
        echo "To monitor:"
        echo "   tail -f logs/vllm_qwen_8bit.log"
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

echo "❌ vLLM server failed to start within 5 minutes"
echo "Check logs: tail -f logs/vllm_qwen_8bit.log"
exit 1

