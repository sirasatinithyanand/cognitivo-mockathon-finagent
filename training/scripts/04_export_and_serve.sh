#!/bin/bash
# Link the trained LoRA adapter and serve via vLLM on port 8001.
# No weight merge required — vLLM loads the adapter at runtime.
#
# Usage:
#   ADAPTER_CHECKPOINT=/models/checkpoints/nemotron8b-finance-lora/finance-lora/checkpoints/last/hf_adapter \
#   bash scripts/04_export_and_serve.sh
#
# After this script completes, restart LiteLLM:
#   docker restart litellm-proxy

set -e

MODEL_DIR="${MODEL_DIR:-/home/cognitivo/local-llm-setup/models}"
BASE_MODEL="${BASE_MODEL:-Llama-3.1-Nemotron-Nano-8B-v1}"
MODEL_NAME="nemotron-8b-finance"
PORT="${PORT:-8001}"
VLLM_IMAGE="vllm/vllm-openai:latest"
GPU_MEM="${GPU_MEM_UTIL:-0.45}"

# Default adapter path — override with ADAPTER_CHECKPOINT env var
ADAPTER_CHECKPOINT="${ADAPTER_CHECKPOINT:-${MODEL_DIR}/checkpoints/nemotron8b-finance-lora/finance-lora/checkpoints/last/hf_adapter}"
ADAPTER_SYMLINK="${MODEL_DIR}/nemotron8b-finance-adapter"

echo "=== Linking LoRA adapter ==="
echo "  Checkpoint: ${ADAPTER_CHECKPOINT}"
echo "  Symlink:    ${ADAPTER_SYMLINK}"

if [ ! -d "${ADAPTER_CHECKPOINT}" ]; then
  echo "ERROR: adapter checkpoint not found at ${ADAPTER_CHECKPOINT}"
  echo "Set ADAPTER_CHECKPOINT to your trained checkpoint directory."
  exit 1
fi

ln -sfn "${ADAPTER_CHECKPOINT}" "${ADAPTER_SYMLINK}"
echo "  Linked."

echo ""
echo "=== Starting vLLM (base + LoRA) on port ${PORT} ==="
docker rm -f vllm-domain-ft 2>/dev/null || true

docker run --gpus all -d \
  --name vllm-domain-ft \
  --restart unless-stopped \
  -p ${PORT}:8000 \
  -v "${MODEL_DIR}":/models \
  "${VLLM_IMAGE}" \
  --model /models/${BASE_MODEL} \
  --served-model-name "${MODEL_NAME}" \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization ${GPU_MEM} \
  --enable-lora \
  --max-lora-rank 32 \
  --lora-modules "${MODEL_NAME}=/models/nemotron8b-finance-adapter" \
  --override-generation-config '{"enable_thinking": false}'

echo "Waiting for server to be ready..."
for i in $(seq 1 60); do
  if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    echo "  Ready at http://localhost:${PORT}/v1  (${i}x5s)"
    break
  fi
  sleep 5
done

echo ""
LITELLM_CONFIG="${LITELLM_CONFIG:-/home/cognitivo/litellm/config.yaml}"
echo "=== Adding alias to LiteLLM config (${LITELLM_CONFIG}) ==="
grep -q "nemotron-8b-finance" "${LITELLM_CONFIG}" 2>/dev/null || cat >> "${LITELLM_CONFIG}" << 'EOF'

  - model_name: nemotron-8b-finance
    litellm_params:
      model: openai/nemotron-8b-finance
      api_base: http://localhost:8001/v1
      api_key: dummy
EOF

echo "Done. Restart LiteLLM to pick up the new alias:"
echo "  docker restart litellm-proxy"
echo ""
echo "Test: curl http://localhost:${PORT}/v1/models"
