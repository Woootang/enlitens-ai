#!/bin/bash
# Start vLLM with Falcon-H1 34B Instruct (GPTQ-int4)
# Optimized for RTX 3090 (24GB VRAM)

set -e

echo "🚀 Starting vLLM with Falcon-H1 34B Instruct"
echo "=============================================="
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
MODEL_PATH="/home/antons-gs/enlitens-ai/models/falcon-h1-34b-instruct-gptq-int4"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model not found at: $MODEL_PATH"
    exit 1
fi

echo "📦 Model: Falcon-H1 34B Instruct (GPTQ-int4)"
echo "📁 Path: $MODEL_PATH"
echo "🔧 Quantization: GPTQ int4 (~17GB VRAM)"
echo "🧠 Context: 256k tokens"
echo ""

# vLLM configuration for Falcon-H1 34B on 24GB GPU
echo "🎯 Starting vLLM server..."
echo "   Port: 8000"
echo "   Max model length: 8192 (model default)"
echo "   GPU memory utilization: 0.90"
echo "   Note: Falcon-H1 supports 256k context, but starting with safe 8k"
echo ""

# Set environment variable to allow longer context if needed
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Start vLLM in background
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port 8000 \
    --host 0.0.0.0 \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --quantization gptq \
    --dtype float16 \
    --trust-remote-code \
    --disable-log-requests \
    > logs/vllm_falcon.log 2>&1 &

VLLM_PID=$!
echo "✅ vLLM started (PID: $VLLM_PID)"
echo "📊 Log: logs/vllm_falcon.log"
echo ""

# Wait for server to be ready
echo "⏳ Waiting for vLLM server to initialize..."
for i in {1..60}; do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "✅ vLLM server is ready!"
        echo ""
        
        # Show loaded model
        echo "📦 Loaded models:"
        curl -s http://localhost:8000/v1/models | python3 -c "import sys,json; [print(f\"   - {m['id']}\") for m in json.load(sys.stdin)['data']]"
        echo ""
        
        echo "🎉 Falcon-H1 34B is ready for PDF processing!"
        echo ""
        echo "To monitor:"
        echo "   tail -f logs/vllm_falcon.log"
        echo ""
        echo "To stop:"
        echo "   pkill -f vllm.entrypoints"
        
        exit 0
    fi
    
    if [ $((i % 10)) -eq 0 ]; then
        echo "   Still loading... ($i seconds)"
    fi
    
    sleep 1
done

echo "❌ vLLM server failed to start within 60 seconds"
echo "Check logs: tail -f logs/vllm_falcon.log"
exit 1

