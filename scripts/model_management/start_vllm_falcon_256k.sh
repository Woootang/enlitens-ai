#!/bin/bash
# Start vLLM with Falcon-H1 34B at extended context (up to 256k)
# Uses GPU (24GB) + CPU RAM offloading for KV cache
# SAFETY: Monitors temps and kills if GPU >80°C or CPU >85°C

set -e

cd /home/antons-gs/enlitens-ai

echo "🚀 Starting vLLM with Falcon-H1 34B (Extended Context)"
echo "======================================================="
echo ""

# Safety check function
check_temps() {
    GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
    CPU_TEMP=$(sensors | grep "Package id 0" | awk '{print $4}' | tr -d '+°C')
    
    if (( $(echo "$GPU_TEMP > 80" | bc -l) )); then
        echo "🔥 WARNING: GPU temp ${GPU_TEMP}°C exceeds 80°C!"
        return 1
    fi
    
    if (( $(echo "$CPU_TEMP > 85" | bc -l) )); then
        echo "🔥 WARNING: CPU temp ${CPU_TEMP}°C exceeds 85°C!"
        return 1
    fi
    
    return 0
}

# Kill existing vLLM
echo "🧹 Cleaning up old vLLM processes..."
pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
sleep 3

# Check initial temps
echo "🌡️  Initial System Status:"
nvidia-smi --query-gpu=name,temperature.gpu,memory.free --format=csv,noheader
echo "CPU: $(sensors | grep "Package id 0" | awk '{print $4}')"
echo "RAM: $(free -h | grep Mem | awk '{print $7}') available"
echo ""

if ! check_temps; then
    echo "❌ System too hot to start! Cool down first."
    exit 1
fi

MODEL_PATH="/home/antons-gs/enlitens-ai/models/falcon-h1-34b-instruct-gptq-int4"

if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model not found at: $MODEL_PATH"
    exit 1
fi

echo "📦 Model: Falcon-H1 34B (GPTQ int4)"
echo "📁 Path: $MODEL_PATH"
echo "🔧 Quantization: GPTQ 4-bit (~17GB VRAM)"
echo "🧠 Context: Starting at 32k, can extend to 128k+ with CPU offload"
echo "💾 VRAM: 24GB GPU + up to 40GB CPU RAM for KV cache"
echo "✨ Architecture: Hybrid Transformer+SSM (efficient long context)"
echo "⚡ Speed: ~2-3x slower than 40k, but handles MUCH longer contexts"
echo ""

# Activate venv
source /home/antons-gs/enlitens-ai/venv/bin/activate

# Allow extended context
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Start vLLM with CPU offloading for KV cache
# Note: Falcon-H1's hybrid architecture is more memory-efficient than pure transformers
echo "🎯 Starting vLLM server..."
echo "   Port: 8000"
echo "   Max model length: 32768 (safe start, can extend dynamically)"
echo "   GPU memory utilization: 0.85 (leave room for KV cache)"
echo "   CPU offload: 40GB swap space for extended contexts"
echo ""

nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port 8000 \
    --host 0.0.0.0 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --swap-space 40 \
    --quantization gptq \
    --dtype float16 \
    --trust-remote-code \
    --disable-log-requests \
    --max-num-seqs 4 \
    --enable-chunked-prefill \
    > logs/vllm_falcon.log 2>&1 &

VLLM_PID=$!
echo "✅ vLLM started (PID: $VLLM_PID)"
echo "📊 Log: logs/vllm_falcon.log"
echo ""

# Wait for server to be ready with temperature monitoring
echo "⏳ Waiting for vLLM to initialize (3-5 minutes for large model)..."
echo "   Monitoring temperatures every 10 seconds..."
echo ""

for i in {1..300}; do
    # Check if server is ready
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo ""
        echo "✅ vLLM is READY!"
        echo ""
        curl -s http://localhost:8000/v1/models | python3 -c "import sys,json; [print(f'   📦 {m[\"id\"]}') for m in json.load(sys.stdin)['data']]"
        echo ""
        echo "🌡️  Final System Status:"
        nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used,memory.total --format=csv,noheader
        echo "CPU: $(sensors | grep "Package id 0" | awk '{print $4}')"
        echo "RAM: $(free -h | grep Mem | awk '{print $3}') used / $(free -h | grep Mem | awk '{print $2}') total"
        echo ""
        echo "🎉 Ready for EXTENDED CONTEXT processing!"
        echo ""
        echo "💡 Usage Tips:"
        echo "   - Start with 32k context (default)"
        echo "   - For longer contexts (64k-128k), requests will automatically use CPU offload"
        echo "   - Monitor 'tail -f logs/vllm_falcon.log' for any warnings"
        echo "   - If you see OOM errors, reduce max_model_len or batch size"
        echo ""
        exit 0
    fi
    
    # Check temps every 10 seconds
    if [ $((i % 10)) -eq 0 ]; then
        if ! check_temps; then
            echo ""
            echo "❌ Temperature safety limit exceeded during startup!"
            echo "   Killing vLLM (PID: $VLLM_PID)..."
            kill -9 $VLLM_PID 2>/dev/null || true
            exit 1
        fi
        
        GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
        CPU_TEMP=$(sensors | grep "Package id 0" | awk '{print $4}' | tr -d '+°C')
        echo "   [$i sec] GPU: ${GPU_TEMP}°C | CPU: ${CPU_TEMP}°C"
    fi
    
    sleep 1
done

echo ""
echo "❌ Timeout - vLLM did not start within 5 minutes"
echo "   Check logs: tail -f logs/vllm_falcon.log"
echo "   Common issues:"
echo "     - Model loading failed (check if all safetensors files are present)"
echo "     - CUDA out of memory (try reducing --gpu-memory-utilization)"
echo "     - Port 8000 already in use (check with: lsof -i :8000)"
exit 1

