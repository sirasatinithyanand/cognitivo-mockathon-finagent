"""Mock LiteLLM — canned OpenAI-compatible `/v1/chat/completions` with scripted tool-calling.

Deterministic brain stand-in so the whole Stage-2..4 loop runs offline:
  round 1 (no tool results in the conversation, tools offered)
      -> parallel tool calls: domain_predict(BHP, 90) + retrieve(...)
  round 2 (tool results present)
      -> final cited answer built from the tool outputs
  no tools offered -> plain canned completion

Run:  python -m mocks.mock_llm         (serves on :9000; config.py's offline default)
"""
from __future__ import annotations

import json
import time
import uuid

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="mock LiteLLM (canned completions)")


class ChatRequest(BaseModel):
    model: str = "agent-brain"
    messages: list[dict]
    tools: list[dict] | None = None
    temperature: float | None = None


def _resp(message: dict, model: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason":
                     "tool_calls" if message.get("tool_calls") else "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.get("/health")
def health():
    return {"status": "ok"}


def _sentiment_reply(prompt: str) -> str | None:
    lower_prompt = prompt.lower()
    if "travel stocks take off on vaccine rollout" in lower_prompt:
        return (
            "The article reads as positive sentiment and points to likely upward momentum for ASX travel shares "
            "as vaccine-led reopening optimism improves the sector outlook."
        )
    if "why investors don't believe the rba on interest rates" in lower_prompt:
        return (
            "The article reads as mixed-to-negative sentiment, with broad ASX sentiment likely mixed to down and "
            "rate-sensitive shares under pressure as investors doubt the RBA's rate guidance."
        )
    if "energy stocks shine as vaccines fuel oil rally" in lower_prompt:
        return (
            "The article reads as positive sentiment and points to likely upward momentum for ASX energy shares "
            "as vaccine-led reopening hopes and oil strength support the sector."
        )
    return None


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    tool_results = [m for m in req.messages if m.get("role") == "tool"]

    if req.tools and not tool_results:
        # round 1: script one parallel tool-call turn (skill + retrieval — DoD #2 shape)
        return _resp({
            "role": "assistant", "content": None,
            "tool_calls": [
                {"id": "call_predict", "type": "function",
                 "function": {"name": "domain_predict",
                              "arguments": json.dumps({"ticker": "BHP", "horizon_days": 90})}},
                {"id": "call_retrieve", "type": "function",
                 "function": {"name": "retrieve",
                              "arguments": json.dumps(
                                  {"query": "BHP results interest rates filings risks", "k": 4,
                                   "filters": {"corpus": "finance"}})}},
            ],
        }, req.model)

    if tool_results:
        # round 2: cite whatever actually came back
        pred, doc_ids = "n/a", []
        for m in tool_results:
            try:
                data = json.loads(m.get("content") or "{}")
            except json.JSONDecodeError:
                continue
            if "prediction" in data:
                pred = f"{data['prediction']}% ({data.get('unit', '')})"
            doc_ids += [d.get("id", "?") for d in data.get("docs", [])]
        cites = ", ".join(f"[{i}]" for i in doc_ids[:4]) or "[no sources]"
        content = (
            f"Based on the Stage-1 model, the 90-day forward return forecast for BHP is {pred}. "
            f"Retrieved filings note iron-ore price sensitivity and refinancing exposure to current "
            f"interest rates {cites}. On balance: HOLD - the forecast is modest and rate risk is "
            f"already priced into the refinancing schedule. Sources: {cites}."
        )
        return _resp({"role": "assistant", "content": content}, req.model)

    prompt = ""
    for message in reversed(req.messages):
        if message.get("role") == "user":
            prompt = message.get("content") or ""
            break
    sentiment_reply = _sentiment_reply(prompt)
    if sentiment_reply:
        return _resp({"role": "assistant", "content": sentiment_reply}, req.model)

    return _resp({"role": "assistant",
                  "content": "Mock brain online. Ask a finance question with tools enabled."},
                 req.model)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9000)
