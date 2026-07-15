#!/bin/bash
# Evaluate base Nemotron-8B vs finetuned 8B LoRA on the held-out test set.
#
# Usage:
#   bash scripts/06_eval_8b.sh
#   bash scripts/06_eval_8b.sh --samples 200   # default 100

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

MODEL_DIR="${MODEL_DIR:-/home/cognitivo/local-llm-setup/models}"
BASE_MODEL="${MODEL_DIR}/Llama-3.1-Nemotron-Nano-8B-v1"
ADAPTER_DIR="${MODEL_DIR}/nemotron8b-finance-adapter"
TEST_FILE="${REPO_DIR}/data/test.jsonl"
OUT_FILE="${REPO_DIR}/eval_8b_report.json"
SAMPLES="${SAMPLES:-100}"

BASE_PORT=8000
FT_PORT=8001
BASE_NAME="nemotron-8b-base"
FT_NAME="nemotron-8b-finance"
FT_HOST="${FT_HOST:-ata1}"   # FT model runs on ata1; override with FT_HOST=localhost for single-node

VLLM_IMAGE="vllm/vllm-openai:latest"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --samples) SAMPLES="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "=== Nemotron-8B Finance LoRA Evaluation ==="
echo "  Base model:  ${BASE_MODEL}"
echo "  LoRA adapter: ${ADAPTER_DIR}"
echo "  Test file:   ${TEST_FILE}"
echo "  Samples:     ${SAMPLES}"
echo ""

# ── Cleanup any stale containers ────────────────────────────────────────────
echo "=== Stopping any existing eval containers ==="
docker rm -f vllm-base-eval 2>/dev/null || true
ssh -T "${FT_HOST}" "docker rm -f vllm-ft-eval 2>/dev/null" || true

# ── Start base model on ata0:8000 ───────────────────────────────────────────
echo ""
echo "=== Starting base model on ata0:${BASE_PORT} ==="
docker run --gpus all -d --rm \
  --name vllm-base-eval \
  -p ${BASE_PORT}:8000 \
  -v "${MODEL_DIR}":/models \
  "${VLLM_IMAGE}" \
  --model /models/Llama-3.1-Nemotron-Nano-8B-v1 \
  --served-model-name "${BASE_NAME}" \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.45 \
  --override-generation-config '{"enable_thinking": false}'

# ── Start finetuned model on ata1:8001 ──────────────────────────────────────
echo "=== Starting finetuned model on ${FT_HOST}:${FT_PORT} ==="
ssh -T "${FT_HOST}" "docker rm -f vllm-ft-eval 2>/dev/null; docker run --gpus all -d --rm \
  --name vllm-ft-eval \
  -p ${FT_PORT}:8000 \
  -v ${MODEL_DIR}:/models \
  ${VLLM_IMAGE} \
  --model /models/Llama-3.1-Nemotron-Nano-8B-v1 \
  --served-model-name ${FT_NAME} \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.45 \
  --enable-lora \
  --max-lora-rank 32 \
  --lora-modules '${FT_NAME}=/models/nemotron8b-finance-adapter' \
  --override-generation-config '{\"enable_thinking\": false}'"

# ── Wait for both endpoints ──────────────────────────────────────────────────
echo ""
echo "=== Waiting for both vLLM servers to be ready (up to 10 min) ==="

wait_for_port() {
  local host=$1 port=$2 label=$3
  for i in $(seq 1 120); do
    if curl -sf "http://${host}:${port}/health" > /dev/null 2>&1; then
      echo "  ${label} ready (${i}x5s)"
      return 0
    fi
    sleep 5
  done
  echo "  ERROR: ${label} did not become ready"
  docker logs vllm-base-eval 2>&1 | tail -10
  ssh -T "${FT_HOST}" "docker logs vllm-ft-eval 2>&1 | tail -10" || true
  docker rm -f vllm-base-eval 2>/dev/null || true
  ssh -T "${FT_HOST}" "docker rm -f vllm-ft-eval 2>/dev/null" || true
  exit 1
}

wait_for_port "localhost" "${BASE_PORT}" "ata0:${BASE_PORT} (base)"
wait_for_port "${FT_HOST}"  "${FT_PORT}"  "${FT_HOST}:${FT_PORT} (ft)"

# ── Run evaluation ───────────────────────────────────────────────────────────
echo ""
echo "=== Running evaluation (${SAMPLES} samples) ==="
python3 "${SCRIPT_DIR}/05_evaluate.py" \
  --test_file  "${TEST_FILE}" \
  --base_url   "http://localhost:${BASE_PORT}/v1" \
  --ft_url     "http://${FT_HOST}:${FT_PORT}/v1" \
  --base_model "${BASE_NAME}" \
  --ft_model   "${FT_NAME}" \
  --max_samples "${SAMPLES}" \
  --out        "${OUT_FILE}"

# ── Cleanup ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Stopping eval containers ==="
docker rm -f vllm-base-eval 2>/dev/null || true
ssh -T "${FT_HOST}" "docker rm -f vllm-ft-eval 2>/dev/null" || true

echo ""
echo "Done. Report: ${OUT_FILE}"
