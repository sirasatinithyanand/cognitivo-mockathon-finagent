#!/bin/bash
# Production LoRA SFT: Qwen3.5-35B-A3B on full 48k finance dataset.
# Single DGX Spark node, BF16, SDPA attention, TRL SFTTrainer.
#
# Run inside tmux so it survives disconnects:
#   tmux new-session -s train "bash scripts/03_train_qwen35.sh"
# Monitor:
#   tail -f /tmp/qwen35_train.log

set -e

NEMO_IMAGE="${NEMO_IMAGE:-nvcr.io/nvidia/nemo:25.09}"
MODEL_DIR="${MODEL_DIR:-/home/cognitivo/local-llm-setup/models}"
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="/tmp/qwen35_train.log"
MAX_STEPS="${MAX_STEPS:-500}"
LORA_RANK="${LORA_RANK:-16}"

echo "=== Qwen3.5-35B-A3B LoRA SFT (MAX_STEPS=${MAX_STEPS}, rank=${LORA_RANK}) ==="
echo "  Workspace:  ${WORKSPACE}"
echo "  Container:  ${NEMO_IMAGE}"
echo "  Log:        ${LOG}"
echo ""

nohup docker run --rm --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  --name qwen35_train \
  -v "${MODEL_DIR}":/models \
  -v "${WORKSPACE}":/workspace \
  -e MODEL_PATH=/models/Qwen3.5-35B-A3B \
  -e TRAIN_FILE=/workspace/data/train.jsonl \
  -e VAL_FILE=/workspace/data/val.jsonl \
  -e OUTPUT_DIR=/models/checkpoints/qwen35-finance-lora \
  -e MAX_STEPS="${MAX_STEPS}" \
  -e MAX_SEQ_LEN=512 \
  -e BATCH_SIZE=1 \
  -e GRAD_ACCUM=4 \
  -e LORA_RANK="${LORA_RANK}" \
  -e LR=1e-4 \
  "${NEMO_IMAGE}" \
  bash -c "
    pip install -q trl 'transformers>=5.0.0' datasets 2>&1 | grep -E 'Successfully|already|ERROR' || true
    python /workspace/training/train_qwen35_trl.py
  " \
  > "${LOG}" 2>&1 &

TRAIN_PID=$!
echo "  Docker PID: ${TRAIN_PID}"
echo "  Monitor:    tail -f ${LOG}"
echo "  Stop:       kill ${TRAIN_PID}  (or: docker stop qwen35_train)"
echo ""

wait ${TRAIN_PID}
echo "=== Training finished ==="
