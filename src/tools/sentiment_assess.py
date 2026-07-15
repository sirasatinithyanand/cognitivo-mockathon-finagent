"""`sentiment_assess` — domain-ft sentiment/technical assessment tool.

Calls the fine-tuned model using the EXACT prompt format it was trained on:

  AFR mode  : Date + RBA rate + AFR Headline + Article → market sentiment
  Technical mode: ASX daily OHLCV + RBA rate → technical + macro assessment

The model outputs structured text assessment — not a number. Use this for
qualitative questions about market tone, sector sentiment, or stock technicals.
"""
from __future__ import annotations

import os

from openai import OpenAI

from agent import config
from tools.query_data import query_data

from .registry import register

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "mode": {
            "type": "string",
            "enum": ["afr", "technical"],
            "description": (
                "'afr': assess sentiment from an AFR article (provide headline + article). "
                "'technical': assess a stock from its daily price data (provide ticker + date)."
            ),
        },
        "date": {
            "type": "string",
            "description": "ISO date string YYYY-MM-DD for the assessment.",
        },
        "headline": {
            "type": "string",
            "description": "[afr mode] AFR article headline.",
        },
        "article": {
            "type": "string",
            "description": "[afr mode] AFR article body text (truncated to ~500 chars is fine).",
        },
        "ticker": {
            "type": "string",
            "description": "[technical mode] ASX ticker e.g. 'BHP.AX'.",
        },
    },
    "required": ["mode", "date"],
}

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "assessment": {"type": "string"},
        "mode": {"type": "string"},
        "date": {"type": "string"},
        "rba_rate": {"type": "number"},
    },
    "required": ["assessment"],
}

_client: OpenAI | None = None

_STOP_PHRASES = [
    "\n\nSentiment Conclusion",
    "\n\nSentiment Rationale",
    "\n\nRisks to monitor:\n\n1.",
    "\n\n1. RBA guidance",
]


def _truncate_assessment(text: str) -> str:
    """Keep only the first complete assessment block matching training format."""
    # Cut at known over-generation phrases
    for phrase in _STOP_PHRASES:
        idx = text.find(phrase)
        if idx > 50:
            text = text[:idx]
    # Fallback: cap at 4 paragraphs (training format uses 3)
    paras = text.strip().split("\n\n")
    return "\n\n".join(paras[:4]).strip()


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=os.environ.get("LITELLM_BASE_URL", config.LITELLM_BASE_URL),
            api_key=os.environ.get("LITELLM_KEY", config.LITELLM_KEY),
        )
    return _client


def _lookup_rba_rate(date: str) -> float | None:
    """Look up the RBA cash rate in effect on a given date."""
    try:
        r = query_data(dataset="rba", metric="lookup_rate", date_from=date, date_to=date)
        return r.get("result")
    except Exception:
        return None


def _build_afr_prompt(date: str, rba_rate: float, headline: str, article: str) -> str:
    return (
        f"Date: {date}\n"
        f"RBA cash rate: {rba_rate}%\n"
        f"AFR Headline: {headline}\n"
        f"Article: {article[:800]}\n\n"
        "As an Australian financial analyst, assess the market sentiment and likely ASX impact."
    )


def _build_technical_prompt(ticker: str, date: str, rba_rate: float) -> str | None:
    """Build technical prompt from ASX price data — returns None if data unavailable."""
    try:
        r = query_data(dataset="asx", metric="lookup_price", ticker=ticker, date_from=date, date_to=date)
        rec = r.get("result", {})
        if not rec:
            return None
        close = rec.get("close", 0)
        high = rec.get("high", close)
        low = rec.get("low", close)
        volume = rec.get("volume", 0)
        daily_range = round((high - low) / close * 100, 2) if close else 0
        return (
            f"ASX daily data for {ticker} on {date}:\n"
            f"  Close: ${close:.2f} | High: ${high:.2f} | Low: ${low:.2f}\n"
            f"  Volume: {int(volume):,} | Daily range: {daily_range}%\n"
            f"  RBA cash rate: {rba_rate}%\n\n"
            f"Provide a technical and macro assessment for {ticker.split('.')[0]}."
        )
    except Exception:
        return None


@register(
    "sentiment_assess",
    (
        "Call the fine-tuned financial domain model for qualitative market assessment. "
        "Use 'afr' mode to assess sentiment from an AFR article (pass headline + article text). "
        "Use 'technical' mode to get a technical + macro read on an ASX stock for a given date. "
        "Returns structured text — not a number. Best for 'what was market sentiment' questions."
    ),
    INPUT_SCHEMA,
    OUTPUT_SCHEMA,
)
def sentiment_assess(
    mode: str,
    date: str,
    headline: str | None = None,
    article: str | None = None,
    ticker: str | None = None,
) -> dict:
    rba_rate = _lookup_rba_rate(date)
    if rba_rate is None:
        return {"error": f"Could not look up RBA rate for {date}"}

    if mode == "afr":
        if not headline or not article:
            return {"error": "afr mode requires headline and article"}
        prompt = _build_afr_prompt(date, rba_rate, headline, article)
    elif mode == "technical":
        if not ticker:
            return {"error": "technical mode requires ticker"}
        prompt = _build_technical_prompt(ticker, date, rba_rate)
        if not prompt:
            return {"error": f"No price data for {ticker} on {date}"}
    else:
        return {"error": f"Unknown mode '{mode}'. Use 'afr' or 'technical'."}

    try:
        resp = _get_client().chat.completions.create(
            model=config.DOMAIN_FT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=250,
        )
        raw = resp.choices[0].message.content.strip()
        # Truncate at the first repeated block (model sometimes loops)
        assessment = _truncate_assessment(raw)
    except Exception as e:
        return {"error": f"domain-ft call failed: {e}"}

    return {
        "assessment": assessment,
        "mode": mode,
        "date": date,
        "rba_rate": rba_rate,
    }
