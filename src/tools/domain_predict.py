"""`domain_predict` — P1's Stage-1 model called as a tool (contracts §2 example, field-for-field).

Backing modes (DOMAIN_PREDICT_MODE):
  mock : deterministic canned value (offline default — lets the whole loop run pre-integration)
  llm  : LiteLLM `domain-ft` alias — post-LoRA this is P1's fine-tuned specialist; pre-LoRA it
         can point at P1's regression shim (both are OpenAI-compatible endpoints)
"""
from __future__ import annotations

import re

from agent import config, llm

from .registry import register

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "ticker": {"type": "string"},
        "horizon_days": {"type": "integer", "minimum": 1},
    },
    "required": ["ticker", "horizon_days"],
}
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "prediction": {"type": "number"},
        "unit": {"type": "string"},
        "sources": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["prediction"],
}


@register("domain_predict",
          "Call the fine-tuned Stage-1 finance model for a domain prediction "
          "(forward percent return for an ASX ticker over a horizon in days).",
          INPUT_SCHEMA, OUTPUT_SCHEMA)
def domain_predict(ticker: str, horizon_days: int) -> dict:
    if config.DOMAIN_PREDICT_MODE == "llm":
        prompt = (f"Predict the {horizon_days}-day forward return in percent for {ticker}. "
                  f"Answer with a single number (the percent return), nothing else.")
        resp = llm.chat([{"role": "user", "content": prompt}], model=config.DOMAIN_FT_MODEL)
        text = resp.choices[0].message.content or ""
        m = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
        pred = float(m.group()) if m else float("nan")
        src = [f"litellm:{config.DOMAIN_FT_MODEL}"]
    else:
        # deterministic pseudo-prediction so offline runs are reproducible and non-trivial
        pred = round(((sum(ord(c) for c in ticker.upper()) % 17) - 8) * 0.35
                     * (horizon_days / 90) ** 0.5, 4)
        src = ["mock:domain_predict"]
    return {"prediction": pred, "unit": "percent_return", "sources": src}
