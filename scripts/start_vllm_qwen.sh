#!/bin/bash
# Start vLLM with Qwen 2.5-14B Instruct (AWQ)
# Optimized for RTX 3090 (24GB VRAM)

set -e

echo "🚀 Starting vLLM with Qwen 2.5-14B Instruct"
echo "============================================"
echo ""

# Kill any existing vLLM processes
echo "🧹 Cleaning up old vLLM processes..."
pkill -f "vllm" 2>/dev/null || true
pkill -f "python.*qwen" 2>/dev/null || true
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
    echo "Available models:"
    ls -1 /home/antons-gs/enlitens-ai/models/ | grep -i qwen
    exit 1
fi

echo "📦 Model: Qwen 2.5-14B Instruct (AWQ)"
echo "📁 Path: $MODEL_PATH"
echo "🔧 Quantization: AWQ (~8GB VRAM)"
echo "🧠 Context: 128k tokens"
echo ""

# Try to start vLLM (if installed)
if command -v vllm &> /dev/null || python3 -c "import vllm" 2>/dev/null; then
    echo "🎯 Starting vLLM server..."
    echo "   Port: 8000"
    echo "   Max model length: 32768"
    echo "   GPU memory utilization: 0.90"
    echo ""
    
    nohup python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --port 8000 \
        --host 0.0.0.0 \
        --tensor-parallel-size 1 \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.90 \
        --quantization awq \
        --dtype auto \
        --trust-remote-code \
        --disable-log-requests \
        > logs/vllm_qwen.log 2>&1 &
    
    VLLM_PID=$!
    echo "✅ vLLM started (PID: $VLLM_PID)"
    echo "📊 Log: logs/vllm_qwen.log"
    
else
    echo "⚠️  vLLM not found, trying Ollama..."
    
    # Check if Ollama is running
    if ! pgrep -f "ollama" > /dev/null; then
        echo "Starting Ollama..."
        nohup ollama serve > logs/ollama.log 2>&1 &
        sleep 3
    fi
    
    # Load model in Ollama
    echo "Loading Qwen in Ollama..."
    ollama run qwen2.5:14b-instruct-q4_K_M &
    sleep 5
fi

echo ""
echo "⏳ Waiting for server to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "✅ Server is ready!"
        echo ""
        curl -s http://localhost:8000/v1/models | python3 -c "import sys,json; [print(f\"   - {m['id']}\") for m in json.load(sys.stdin)['data']]" 2>/dev/null || echo "   - Model loaded"
        echo ""
        echo "🎉 Qwen 2.5-14B is ready for PDF processing!"
        exit 0
    fi
    
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama is ready!"
        echo ""
        echo "🎉 Qwen 2.5-14B is ready for PDF processing!"
        exit 0
    fi
    
    if [ $((i % 10)) -eq 0 ]; then
        echo "   Still loading... ($i seconds)"
    fi
    
    sleep 1
done

echo "❌ Server failed to start within 60 seconds"
echo "Check logs: tail -f logs/vllm_qwen.log logs/ollama.log"
exit 1

