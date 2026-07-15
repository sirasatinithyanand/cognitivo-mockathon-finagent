#!/bin/bash
# Smoke test: Qwen3.5-35B-A3B LoRA, 20 steps, smoke data (500 samples).
# Validates model load, SDPA attention, LoRA wiring, and first training steps.
# Expected runtime: ~5-10 min (model load ~3 min, 20 steps ~2-5 min).
#
# Run:
#   bash scripts/02_smoke_qwen35.sh
# Monitor:
#   tail -f /tmp/smoke_qwen35.log

set -e

NEMO_IMAGE="${NEMO_IMAGE:-nvcr.io/nvidia/nemo:25.09}"
MODEL_DIR="${MODEL_DIR:-/home/cognitivo/local-llm-setup/models}"
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="/tmp/smoke_qwen35.log"

echo "=== Smoke test: Qwen3.5-35B-A3B LoRA, 20 steps ==="
echo "  Workspace:  ${WORKSPACE}"
echo "  Container:  ${NEMO_IMAGE}"
echo "  Log:        ${LOG}"
echo ""

docker run --rm --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  -v "${MODEL_DIR}":/models \
  -v "${WORKSPACE}":/workspace \
  -e MODEL_PATH=/models/Qwen3.5-35B-A3B \
  -e TRAIN_FILE=/workspace/data/smoke/train.jsonl \
  -e VAL_FILE=/workspace/data/smoke/val.jsonl \
  -e OUTPUT_DIR=/models/checkpoints/smoke-qwen35-lora \
  -e MAX_STEPS=20 \
  -e MAX_SEQ_LEN=512 \
  -e BATCH_SIZE=1 \
  -e GRAD_ACCUM=1 \
  -e LORA_RANK=8 \
  -e LR=2e-4 \
  "${NEMO_IMAGE}" \
  bash -c "
    pip install -q trl 'transformers>=5.0.0' datasets 2>&1 | grep -E 'Successfully|already|ERROR' || true
    python /workspace/train_qwen35_trl.py
  " \
  2>&1 | tee "${LOG}"

echo "=== Smoke test finished ==="
