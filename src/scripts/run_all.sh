#!/usr/bin/env bash
# Run all 15 public questions through the agent on the real Qwen brain and log results.
set -uo pipefail
cd "$(dirname "$0")/.."
export DATA_DIR="/home/cognitivo/Desktop/HackathonDataset/Jsonl format DataSets"
export LITELLM_BASE_URL="http://10.0.1.10:4000/v1"
export LITELLM_KEY="EMPTY"
export BRAIN_MODEL="agent-brain"
export DOMAIN_FT_MODEL="domain-ft"
export DOMAIN_PREDICT_MODE="llm"
export MAX_AGENT_STEPS=6
export PUBLIC_QUESTIONS="/home/cognitivo/Cognitivo_Training/Mock_Hackathon_Participant_Package/public_questions.jsonl"
/home/cognitivo/Desktop/.venv/bin/python -m scripts.check_questions "$@"
