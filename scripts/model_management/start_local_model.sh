#!/usr/bin/env bash
# Usage: ./scripts/start_local_model.sh medgemma
set -euo pipefail
MODEL_KEY="${1:?model key required (medgemma|llama)}"
source "$(dirname "$0")/../venv/bin/activate"
MAX_LEN=131072
case "$MODEL_KEY" in
  medgemma)
    MODEL_PATH="models/medgemma-4b-it"
    MAX_LEN=131072
    KV_DTYPE="auto"
    ;;
  llama)
    MODEL_PATH="models/llama-3.1-8b-instruct"
    MAX_LEN=58000
    KV_DTYPE="auto"
    ;;
  *)
    echo "Unknown model key: $MODEL_KEY" >&2
    exit 1
    ;;
esac
LOG_DIR="logs/local_models"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$MODEL_KEY-$(date +%Y%m%d_%H%M%S).log"
echo "Starting vLLM for $MODEL_KEY -> $MODEL_PATH"
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization 0.95 \
  --swap-space 20 \
  --max-num-seqs 1 \
  --kv-cache-dtype "$KV_DTYPE" \
  --enforce-eager \
  --enable-prefix-caching >"$LOG_FILE" 2>&1 &
PID=$!
echo $PID > "logs/${MODEL_KEY}.pid"
echo "vLLM started (PID=$PID, log=$LOG_FILE)"

