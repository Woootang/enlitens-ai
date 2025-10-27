#!/bin/bash
# Fix Ollama GPU Configuration

echo "🔧 FIXING OLLAMA GPU CONFIGURATION"
echo "=================================="

# Create systemd override directory
echo "📁 Creating systemd override directory..."
sudo mkdir -p /etc/systemd/system/ollama.service.d/

# Create GPU configuration override
echo "⚙️ Creating GPU configuration..."
sudo tee /etc/systemd/system/ollama.service.d/gpu.conf > /dev/null << 'EOF'
[Service]
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="OLLAMA_NUM_GPU=1"
Environment="OLLAMA_GPU_OVERHEAD=0"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=f16"
Environment="OLLAMA_NUM_CTX=4096"
EOF

echo "✅ GPU configuration created"

# Reload systemd
echo "🔄 Reloading systemd..."
sudo systemctl daemon-reload

# Restart Ollama
echo "🔄 Restarting Ollama service..."
sudo systemctl restart ollama

# Wait for Ollama to start
echo "⏳ Waiting for Ollama to start..."
sleep 5

# Check status
echo "📊 Checking Ollama status..."
sudo systemctl status ollama --no-pager | head -20

echo ""
echo "✅ OLLAMA GPU CONFIGURATION COMPLETE"
echo ""
echo "🧪 Testing GPU usage..."
echo "Run this command to test: ollama run qwen3:32b 'Say hello'"
echo ""
echo "📊 Monitor GPU with: watch -n 1 nvidia-smi"

