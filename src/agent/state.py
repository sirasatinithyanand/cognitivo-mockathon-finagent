"""Agent state carried through the LangGraph nodes."""
from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    messages: list[dict]        # OpenAI-format conversation (system/user/assistant/tool)
    sources: list[dict]         # accumulated Doc payloads from retrieval (contracts §3 shape)
    tool_trace: list[dict]      # {name, args, ms, summary} per call — judges' workflow evidence
    steps: int                  # loop guard against runaway tool-calling
    answer: str                 # final assistant text
    extra: dict[str, Any]       # scratch space (kept for forward-compat)
