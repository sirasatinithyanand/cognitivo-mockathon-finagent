#!/bin/bash
# Quick 100-step 8B LoRA test on ata1 (ata0 stays free for brain serving).
# Checkpoints every 20 steps; GRAD_ACCUM=4 (half the 2-node default).
# Expected wall time: ~2-3 hrs.
#
# Run from ata0 inside tmux:
#   tmux new-session -s train8b "bash scripts/07_train_8b_quicktest.sh"
#
# Monitor on ata1:
#   ssh ata1 tail -f /tmp/nemo_8b_test.log
# Stop:
#   ssh ata1 "docker stop nemo_8b_test"

set -e

NEMO_IMAGE="${NEMO_IMAGE:-nvcr.io/nvidia/nemo:25.09}"
MODEL_DIR="${MODEL_DIR:-/home/cognitivo/local-llm-setup/models}"
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ATA1="/tmp/nemo_8b_test.log"

MODEL_PATH="/models/Llama-3.1-Nemotron-Nano-8B-v1"
OUTPUT_DIR="/models/checkpoints/nemotron8b-finance-lora-v3"
MAX_STEPS=100
LORA_RANK=32
GRAD_ACCUM=4
LR="5e-5"

echo "=== Syncing workspace to ata1 ==="
tar czf - -C "$WORKSPACE" data training scripts \
  | ssh -T ata1 "mkdir -p $WORKSPACE && tar xzf - -C $WORKSPACE"

echo "=== Launching 8B 100-step test on ata1 ==="
echo "  Model:      ${MODEL_PATH}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  Steps:      ${MAX_STEPS}  GradAccum:${GRAD_ACCUM}  LoRA rank:${LORA_RANK}"
echo ""

ssh -T ata1 "nohup docker run --rm --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  --name nemo_8b_test \
  -v ${MODEL_DIR}:/models \
  -v ${WORKSPACE}:/workspace \
  -e MODEL_PATH=${MODEL_PATH} \
  -e OUTPUT_DIR=${OUTPUT_DIR} \
  -e MAX_STEPS=${MAX_STEPS} \
  -e LORA_RANK=${LORA_RANK} \
  -e GRAD_ACCUM=${GRAD_ACCUM} \
  -e LR=${LR} \
  ${NEMO_IMAGE} \
  python /workspace/train_1node.py \
  > ${LOG_ATA1} 2>&1 &
echo \"nemo_8b_test started (PID \$!)\"
echo \"Monitor: tail -f ${LOG_ATA1}\""

echo ""
echo "=== ata1 training launched ==="
echo "  Monitor:  ssh ata1 tail -f ${LOG_ATA1}"
echo "  Stop:     ssh ata1 'docker stop nemo_8b_test'"
echo ""
echo "  ata0 is free — start brain serving separately:"
echo "    bash scripts/08_serve_brain.sh"
