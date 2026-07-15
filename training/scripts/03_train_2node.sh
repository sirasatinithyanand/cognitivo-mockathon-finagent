#!/bin/bash
# 2-node NeMo 2.x LoRA SFT with FSDP2 across ata0 + ata1.
# Requires nemo:25.09 — ships NCCL 2.27 which is GB10-aware.
# nemo:25.04 with NCCL 2.25 crashes on first kernel launch on GB10.
#
# Run inside tmux on ata0:
#   tmux new-session -s finetune "bash scripts/03_train_2node.sh"
#
# Monitor:
#   tail -f /tmp/nemo_ata0.log
#   ssh ata1 tail -f /tmp/nemo_ata1.log

set -e

MASTER_ADDR="${MASTER_ADDR:-10.0.0.10}"   # set to ata0's IP on your cluster
MASTER_PORT="${MASTER_PORT:-29500}"
NEMO_IMAGE="${NEMO_IMAGE:-nvcr.io/nvidia/nemo:25.09}"
MODEL_DIR="${MODEL_DIR:-/home/cognitivo/local-llm-setup/models}"
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ATA0="/tmp/nemo_ata0.log"
LOG_ATA1="/tmp/nemo_ata1.log"
PYTHON_SCRIPT="/workspace/training/train_2node.py"

MODEL_PATH="${MODEL_PATH:-/models/Llama-3.1-Nemotron-Nano-8B-v1}"
OUTPUT_DIR="${OUTPUT_DIR:-/models/checkpoints/nemotron8b-finance-lora}"
MAX_STEPS="${MAX_STEPS:-100}"
LORA_RANK="${LORA_RANK:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-5e-5}"

echo "=== Syncing workspace to ata1 ==="
tar czf - -C "$WORKSPACE" data training scripts | ssh -T ata1 "mkdir -p $WORKSPACE && tar xzf - -C $WORKSPACE"

# NCCL env — TCP over CX7 (IB disabled: /dev/infiniband/* not mounted in container)
NCCL_ENV="\
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 \
  -e NCCL_TIMEOUT=1800 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e NCCL_DEBUG=INFO \
  -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  -e TP_SOCKET_IFNAME=enp1s0f0np0"

DOCKER_COMMON="docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --oom-score-adj=-1000 \
  --network host -v ${MODEL_DIR}:/models -v ${WORKSPACE}:/workspace ${NCCL_ENV}"

# --rdzv_backend=static: rank 0 creates TCPStore on MASTER_ADDR:MASTER_PORT; other ranks connect
TORCHRUN_COMMON="torchrun --nnodes=2 --nproc_per_node=1 --rdzv_backend=static \
  --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT}"

echo "=== Starting ata0 (rank 0) ==="
echo "  MODEL:   ${MODEL_PATH}"
echo "  OUTPUT:  ${OUTPUT_DIR}"
echo "  STEPS:   ${MAX_STEPS}  RANK: ${LORA_RANK}  GRAD_ACCUM: ${GRAD_ACCUM}  LR: ${LR}"
nohup ${DOCKER_COMMON} \
  --name nemo_train_ata0 \
  -e MASTER_ADDR=${MASTER_ADDR} \
  -e MASTER_PORT=${MASTER_PORT} \
  -e WORLD_SIZE=2 \
  -e NODE_RANK=0 \
  -e MODEL_PATH=${MODEL_PATH} \
  -e OUTPUT_DIR=${OUTPUT_DIR} \
  -e MAX_STEPS=${MAX_STEPS} \
  -e LORA_RANK=${LORA_RANK} \
  -e GRAD_ACCUM=${GRAD_ACCUM} \
  -e LR=${LR} \
  ${NEMO_IMAGE} \
  ${TORCHRUN_COMMON} --node_rank=0 ${PYTHON_SCRIPT} \
  > "${LOG_ATA0}" 2>&1 &

ATA0_PID=$!
echo "  ata0 PID ${ATA0_PID}, log: ${LOG_ATA0}"

echo "=== Waiting for ata0 to open port ${MASTER_PORT} (up to 5 min) ==="
for i in $(seq 1 60); do
  if nc -z ${MASTER_ADDR} ${MASTER_PORT} 2>/dev/null; then
    echo "  ata0 port ${MASTER_PORT} is listening (took ${i}x5s)"
    break
  fi
  if ! kill -0 ${ATA0_PID} 2>/dev/null; then
    echo "  ERROR: ata0 exited before opening port — check ${LOG_ATA0}"
    exit 1
  fi
  sleep 5
done

echo "=== Starting ata1 (rank 1) ==="
ssh -T ata1 "nohup ${DOCKER_COMMON} \
  --name nemo_train_ata1 \
  -e MASTER_ADDR=${MASTER_ADDR} \
  -e MASTER_PORT=${MASTER_PORT} \
  -e WORLD_SIZE=2 \
  -e NODE_RANK=1 \
  -e MODEL_PATH=${MODEL_PATH} \
  -e OUTPUT_DIR=${OUTPUT_DIR} \
  -e MAX_STEPS=${MAX_STEPS} \
  -e LORA_RANK=${LORA_RANK} \
  -e GRAD_ACCUM=${GRAD_ACCUM} \
  -e LR=${LR} \
  ${NEMO_IMAGE} \
  ${TORCHRUN_COMMON} --node_rank=1 ${PYTHON_SCRIPT} \
  > ${LOG_ATA1} 2>&1 &"
echo "  ata1 started. Monitor: ssh ata1 tail -f ${LOG_ATA1}"

echo ""
echo "=== Both nodes running ==="
echo "  Stop:    kill ${ATA0_PID}; ssh ata1 'docker stop nemo_train_ata1'"
echo "  Monitor: tail -f ${LOG_ATA0}"

wait ${ATA0_PID}
echo "=== ata0 finished. Check ata1: ssh ata1 tail -30 ${LOG_ATA1} ==="
