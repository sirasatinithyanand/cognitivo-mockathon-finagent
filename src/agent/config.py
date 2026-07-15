"""Environment-driven configuration — the only place endpoints/aliases are read.

Offline defaults point at mocks/; flipping these env vars to the real stack (contracts §1
aliases through LiteLLM) is the entire integration step. No code changes.
"""
from __future__ import annotations

import os

LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://localhost:9000/v1")  # mock default
LITELLM_KEY = os.environ.get("LITELLM_KEY", "EMPTY")

BRAIN_MODEL = os.environ.get("BRAIN_MODEL", "agent-brain")
DOMAIN_FT_MODEL = os.environ.get("DOMAIN_FT_MODEL", "domain-ft")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "embed")

# RAG backend: unset -> fixture fallback (mocks/fixtures/docs.json)
QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "sample5")

# domain_predict backing: "mock" (deterministic canned) | "llm" (LiteLLM domain-ft alias)
DOMAIN_PREDICT_MODE = os.environ.get("DOMAIN_PREDICT_MODE", "mock")

# Path to raw JSONL datasets (RBA, ASX, AFR)
DATA_DIR = os.environ.get(
    "DATA_DIR",
    "/home/cognitivo/Downloads/Jasonl format DataSets"
)

MAX_AGENT_STEPS = int(os.environ.get("MAX_AGENT_STEPS", "6"))
