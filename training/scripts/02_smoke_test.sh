#!/bin/bash
# Smoke test: single-node LoRA finetune of Llama-3.2-1B, 50 steps, 500 samples.
# Validates the full NeMo pipeline in ~5 min without triggering earlyoom.
#
# Run:
#   bash scripts/02_smoke_test.sh

set -e

NEMO_IMAGE="${NEMO_IMAGE:-nvcr.io/nvidia/nemo:25.09}"
MODEL_DIR="${MODEL_DIR:-/home/cognitivo/local-llm-setup/models}"
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="/tmp/smoke_1node.log"

echo "=== Smoke test: Llama-3.2-1B LoRA, 50 steps, 500 samples ==="
echo "  Workspace:  ${WORKSPACE}"
echo "  Container:  ${NEMO_IMAGE}"
echo "  Log:        ${LOG}"
echo ""

docker run --rm --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  -v "${MODEL_DIR}":/models \
  -v "${WORKSPACE}":/workspace \
  -e MODEL_PATH=/models/Llama-3.2-1B-Instruct \
  -e TRAIN_FILE=/workspace/data/smoke/train.jsonl \
  -e VAL_FILE=/workspace/data/smoke/val.jsonl \
  -e OUTPUT_DIR=/models/checkpoints/smoke-1b-lora \
  -e MAX_STEPS=50 \
  -e MAX_SEQ_LEN=512 \
  -e BATCH_SIZE=1 \
  -e GRAD_ACCUM=1 \
  -e LORA_RANK=8 \
  -e LR=2e-4 \
  "${NEMO_IMAGE}" \
  python /workspace/train_1node.py \
  2>&1 | tee "${LOG}"

echo "=== Smoke test finished ==="
