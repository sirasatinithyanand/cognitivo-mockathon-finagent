#!/bin/bash
# 2-node smoke test: Llama-3.2-1B, FSDP2, 50 steps.
# Validates NCCL/torchrun, FSDP2Strategy, LoRA wiring, and first training steps
# across ata0 + ata1 before attempting the 49B production run.
#
# Run on ata0 (inside tmux recommended):
#   bash scripts/02_smoke_2node.sh
#
# Monitor:
#   tail -f /tmp/smoke2node_ata0.log
#   ssh ata1 tail -f /tmp/smoke2node_ata1.log

set -e

MASTER_ADDR="10.0.0.10"
MASTER_PORT=29501
NEMO_IMAGE="${NEMO_IMAGE:-nvcr.io/nvidia/nemo:25.09}"
MODEL_DIR="${MODEL_DIR:-/home/cognitivo/local-llm-setup/models}"
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ATA0="/tmp/smoke2node_ata0.log"
LOG_ATA1="/tmp/smoke2node_ata1.log"
PYTHON_SCRIPT="/workspace/train_2node.py"

echo "=== 2-node smoke test: Llama-3.2-1B, FSDP2, 50 steps ==="
echo "  Workspace:  ${WORKSPACE}"
echo "  Container:  ${NEMO_IMAGE}"
echo "  Logs:       ${LOG_ATA0}  /  ssh ata1 tail -f ${LOG_ATA1}"
echo ""

echo "=== Syncing workspace to ata1 ==="
tar czf - -C "$WORKSPACE" data training scripts | ssh -T ata1 "mkdir -p $WORKSPACE && tar xzf - -C $WORKSPACE"
echo "  Sync done."

# NCCL env — TCP over CX7 (IB not mounted in container)
NCCL_ENV="\
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 \
  -e NCCL_TIMEOUT=1800 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e NCCL_DEBUG=INFO \
  -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  -e TP_SOCKET_IFNAME=enp1s0f0np0"

DOCKER_COMMON="docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host -v ${MODEL_DIR}:/models -v ${WORKSPACE}:/workspace ${NCCL_ENV}"

TORCHRUN_COMMON="torchrun --nnodes=2 --nproc_per_node=1 --rdzv_backend=static \
  --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT}"

MODEL_ENV="\
  -e MODEL_PATH=/models/Llama-3.1-Nemotron-Nano-8B-v1 \
  -e TRAIN_FILE=/workspace/data/smoke/train.jsonl \
  -e VAL_FILE=/workspace/data/smoke/val.jsonl \
  -e OUTPUT_DIR=/models/checkpoints/smoke-nemotron8b-lora \
  -e MAX_STEPS=50 \
  -e MAX_SEQ_LEN=512 \
  -e BATCH_SIZE=1 \
  -e GRAD_ACCUM=1 \
  -e LORA_RANK=8 \
  -e LR=2e-4"

echo "=== Starting ata0 (rank 0) ==="
nohup ${DOCKER_COMMON} \
  --name smoke2node_ata0 \
  -e MASTER_ADDR=${MASTER_ADDR} \
  -e MASTER_PORT=${MASTER_PORT} \
  -e WORLD_SIZE=2 \
  -e NODE_RANK=0 \
  ${MODEL_ENV} \
  ${NEMO_IMAGE} \
  ${TORCHRUN_COMMON} --node_rank=0 ${PYTHON_SCRIPT} \
  > "${LOG_ATA0}" 2>&1 &

ATA0_PID=$!
echo "  ata0 PID ${ATA0_PID}, log: ${LOG_ATA0}"

echo "=== Waiting for ata0 to open port ${MASTER_PORT} (up to 3 min) ==="
for i in $(seq 1 36); do
  if nc -z ${MASTER_ADDR} ${MASTER_PORT} 2>/dev/null; then
    echo "  ata0 port ${MASTER_PORT} is listening (took $((i*5))s)"
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
  --name smoke2node_ata1 \
  -e MASTER_ADDR=${MASTER_ADDR} \
  -e MASTER_PORT=${MASTER_PORT} \
  -e WORLD_SIZE=2 \
  -e NODE_RANK=1 \
  ${MODEL_ENV} \
  ${NEMO_IMAGE} \
  ${TORCHRUN_COMMON} --node_rank=1 ${PYTHON_SCRIPT} \
  > ${LOG_ATA1} 2>&1 &"
echo "  ata1 started."

echo ""
echo "=== Both nodes running ==="
echo "  ata0 log: tail -f ${LOG_ATA0}"
echo "  ata1 log: ssh ata1 tail -f ${LOG_ATA1}"
echo "  Stop:     kill ${ATA0_PID}; ssh ata1 'docker stop smoke2node_ata1'"
echo ""

wait ${ATA0_PID}
STATUS=$?
if [ $STATUS -eq 0 ]; then
  echo "=== Smoke test PASSED (ata0 exited 0). Check ata1: ssh ata1 tail -30 ${LOG_ATA1} ==="
else
  echo "=== Smoke test FAILED (ata0 exit ${STATUS}). Logs: ${LOG_ATA0} ==="
  exit $STATUS
fi
