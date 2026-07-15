#!/bin/bash
# Serve Nemotron-8B + LoRA adapter as domain-ft on ata1:8001.
# Thinking mode explicitly disabled — model fine-tuned on direct financial outputs.
#
# Run on ata1 (or via: ssh ata1 "bash ..." ):
#   bash scripts/08_serve_domain_ft.sh
#
# LiteLLM alias: domain-ft → http://ata1:8001/v1

set -e

MODEL_DIR="${MODEL_DIR:-/home/cognitivo/local-llm-setup/models}"
BASE_NAME="nemotron-8b-base"  # internal-only; nothing external calls this name
MODEL_NAME="domain-ft"        # LoRA alias — matches submission.json's model_name for
                               # direct technical review at this endpoint, and is what
                               # LiteLLM's litellm_params.model must also reference
PORT="${PORT:-8001}"
VLLM_IMAGE="vllm/vllm-openai:latest"
GPU_MEM="${GPU_MEM_UTIL:-0.45}"

echo "=== Serving domain-ft (Nemotron-8B + LoRA) on port ${PORT} ==="
echo "  Adapter:  ${MODEL_DIR}/nemotron8b-finance-adapter"
echo "  Thinking: OFF (enable_thinking=false)"
echo ""

docker rm -f vllm-domain-ft 2>/dev/null || true

# NOTE: --served-model-name and the --lora-modules name must be DISTINCT.
# Registering both under the same string is ambiguous about which one a
# request for that name actually resolves to; use a separate internal name
# for the base model and keep MODEL_NAME solely as the LoRA alias.
docker run --gpus all -d \
  --name vllm-domain-ft \
  --restart unless-stopped \
  -p ${PORT}:8000 \
  -v "${MODEL_DIR}":/models \
  "${VLLM_IMAGE}" \
  --model /models/Llama-3.1-Nemotron-Nano-8B-v1 \
  --served-model-name "${BASE_NAME}" \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization ${GPU_MEM} \
  --enable-lora \
  --max-lora-rank 32 \
  --lora-modules "${MODEL_NAME}=/models/nemotron8b-finance-adapter" \
  --override-generation-config '{"enable_thinking": false}'

echo "Waiting for domain-ft to be ready..."
for i in $(seq 1 60); do
  if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    echo "  Ready at http://localhost:${PORT}/v1  (${i}x5s)"
    break
  fi
  sleep 5
done

echo ""
echo "Test: curl http://localhost:${PORT}/v1/models"
