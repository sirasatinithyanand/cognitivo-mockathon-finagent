"""OpenAI-SDK client factory. All LLM traffic goes through the LiteLLM proxy (contracts §1) —
never call vLLM hosts directly from agent code."""
from __future__ import annotations

import os

from openai import OpenAI

from . import config


def get_client() -> OpenAI:
    return OpenAI(base_url=config.LITELLM_BASE_URL, api_key=config.LITELLM_KEY)


def chat(messages: list[dict], tools: list[dict] | None = None, model: str | None = None):
    """One chat-completions call against the brain (or an explicit alias)."""
    max_tokens = int(os.environ.get("BRAIN_MAX_TOKENS", "1024"))
    max_tool_calls = int(os.environ.get("MAX_AGENT_STEPS", "6"))

    # Count how many tool results are already in the conversation.
    tool_count = sum(1 for m in messages if m.get("role") == "tool")

    if tools and tool_count < max_tool_calls:
        active_tools = tools
        tool_choice = "required"
    else:
        active_tools = None
        tool_choice = None

    return get_client().chat.completions.create(
        model=model or config.BRAIN_MODEL,
        messages=messages,
        tools=active_tools,
        tool_choice=tool_choice,
        temperature=0.1,
        max_tokens=max_tokens,
    )
