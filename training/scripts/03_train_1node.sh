#!/bin/bash
# Single-node NeMo 2.x LoRA SFT on ata0 (SingleDeviceStrategy).
# Uses Llama-3_3-Nemotron-Super-49B-v1 — requires nemo:25.09+ for GB10 support.
#
# Run inside tmux so it survives disconnects:
#   tmux new-session -s finetune "bash scripts/03_train_1node.sh"
#
# Monitor:
#   tail -f /tmp/nemo_1node.log

set -e

NEMO_IMAGE="${NEMO_IMAGE:-nvcr.io/nvidia/nemo:25.09}"
MODEL_DIR="${MODEL_DIR:-/home/cognitivo/local-llm-setup/models}"
MODEL_PATH="${MODEL_PATH:-/models/Llama-3.1-Nemotron-Nano-8B-v1}"
OUTPUT_DIR="${OUTPUT_DIR:-/models/checkpoints/nemotron8b-finance-lora}"
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="/tmp/nemo_1node.log"
MAX_STEPS="${MAX_STEPS:-100}"
LORA_RANK="${LORA_RANK:-32}"
LR="${LR:-5e-5}"

echo "=== Single-node LoRA finetune — Nemotron-8B (MAX_STEPS=${MAX_STEPS}) ==="
echo "  Workspace:  ${WORKSPACE}"
echo "  Container:  ${NEMO_IMAGE}"
echo "  Log:        ${LOG}"
echo ""

nohup docker run --rm --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  --name nemo_train_prod \
  -v "${MODEL_DIR}":/models \
  -v "${WORKSPACE}":/workspace \
  -e MODEL_PATH="${MODEL_PATH}" \
  -e OUTPUT_DIR="${OUTPUT_DIR}" \
  -e MAX_STEPS="${MAX_STEPS}" \
  -e LORA_RANK="${LORA_RANK}" \
  -e LR="${LR}" \
  "${NEMO_IMAGE}" \
  python /workspace/train_1node.py \
  > "${LOG}" 2>&1 &

TRAIN_PID=$!
echo "  Docker PID: ${TRAIN_PID}"
echo "  Monitor:    tail -f ${LOG}"
echo "  Stop:       kill ${TRAIN_PID}"
echo ""

wait ${TRAIN_PID}
echo "=== Training finished ==="
