#!/usr/bin/env bash
# Stable orchestration script for the Enlitens processing pipeline.
set -euo pipefail

VLLM_MAIN_MODEL="qwen2.5-32b-instruct-q4_k_m"
VLLM_MONITOR_MODEL="qwen2.5-3b-instruct-q4_k_m"
VLLM_GPU_UTIL="0.92"
MAIN_PORT="8000"
MONITOR_PORT="8001"
LOG_DIR="logs"

start_vllm_server() {
  local model="$1"
  local port="$2"
  local log_file="$3"
  local extra_flags=("--gpu-memory-utilization" "$VLLM_GPU_UTIL" "--max-num-seqs" "24" "--enforce-eager")

  if ! pgrep -f "vllm.*--port ${port}" >/dev/null 2>&1; then
    echo "🚀 Starting vLLM server for ${model} on port ${port}"
    nohup python3 -m vllm.entrypoints.openai.api_server \
      --model "${model}" \
      --quantization "q4_k_m" \
      --dtype "auto" \
      --tensor-parallel-size 1 \
      --port "${port}" \
      --host "0.0.0.0" \
      --enable-paged-attention \
      --enable-chunked-prefill \
      --kv-cache-dtype auto \
      "${extra_flags[@]}" \
      > "${log_file}" 2>&1 &
    sleep 5
    if ! pgrep -f "vllm.*--port ${port}" >/dev/null 2>&1; then
      echo "❌ Failed to start vLLM server for ${model}"
      exit 1
    fi
  else
    echo "✅ vLLM server already running on port ${port}"
  fi
}

mkdir -p "${LOG_DIR}"

echo "🚀 ENLITENS MULTI-AGENT PROCESSING SYSTEM"
echo "=========================================="
echo

echo "🧹 Cleaning up old logs and outputs"
rm -f *.json *.json.temp *.log || true
rm -f ${LOG_DIR}/*.log || true

start_vllm_server "${VLLM_MAIN_MODEL}" "${MAIN_PORT}" "${LOG_DIR}/vllm-main.log"
start_vllm_server "${VLLM_MONITOR_MODEL}" "${MONITOR_PORT}" "${LOG_DIR}/vllm-monitor.log"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "🔥 GPU Status:"
  nvidia-smi --query-gpu=name,memory.total,memory.used,temperature.gpu --format=csv,noheader
else
  echo "⚠️ nvidia-smi not available"
fi

echo
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="enlitens_knowledge_base_complete_${TIMESTAMP}.json"
TEMP_LOG="${LOG_DIR}/temp_processing.log"

echo "🎯 Launching enhanced multi-agent processor..."
nohup python3 process_multi_agent_corpus.py \
  --input-dir enlitens_corpus/input_pdfs \
  --output-file "${OUTPUT_FILE}" \
  --st-louis-report st_louis_health_report.pdf \
  > "${TEMP_LOG}" 2>&1 &
PROCESS_PID=$!

echo "✅ Process started with PID: ${PROCESS_PID}"
echo "📄 Output target: ${OUTPUT_FILE}"
echo "🪵 Live log: ${TEMP_LOG}"
