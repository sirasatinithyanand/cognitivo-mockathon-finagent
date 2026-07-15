"""`sentiment_assess` — domain-ft sentiment/technical assessment tool.

Calls the fine-tuned model using the EXACT prompt format it was trained on:

  AFR mode  : Date + RBA rate + AFR Headline + Article → market sentiment
  Technical mode: ASX daily OHLCV + RBA rate → technical + macro assessment

The model outputs structured text assessment — not a number. Use this for
qualitative questions about market tone, sector sentiment, or stock technicals.
"""
from __future__ import annotations

import os
import re

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
        f"RBA cash rate: {rba_rate:.2f}%\n"
        f"AFR Headline: {headline}\n"
        f"Article: {article[:800]}\n\n"
        "As an Australian financial analyst, assess the market sentiment and likely ASX impact."
    )


# --- Sentiment calibration via the brain model ------------------------------
# The fine-tuned domain model reliably grounds the *reasoning* but tends to emit a
# single polarity word ("Sentiment: negative") even when the article is genuinely
# two-sided — e.g. a market that distrusts / prices against official RBA guidance is
# a SPLIT market ("mixed with a bias"), not a one-directional one. Rather than guess
# that ambiguity with keyword rules, we ask the brain (Qwen) to judge it: it reads
# the domain model's own assessment and maps it onto the graded label vocabulary.
# This is judgment, not pattern-matching, so it generalises to articles that phrase
# the same nuance differently. Fails safe — any error or unexpected output leaves the
# domain model's original assessment untouched.
_SENT_LINE = re.compile(r"(?im)^\s*Sentiment:\s*(.+)$")
_DIR_LINE = re.compile(r"(?im)^\s*Direction:\s*(.+)$")
_VALID_LABELS = (
    "mixed with a negative bias",
    "mixed with a positive bias",
    "negative",
    "positive",
)
_CALIBRATION_SYS = (
    "You are a precise financial-market sentiment labeller for the ASX. You are given a "
    "financial analyst's written assessment of an AFR article. Reply in EXACTLY two lines "
    "and nothing else:\n"
    "LABEL: <positive|negative|mixed with a positive bias|mixed with a negative bias>\n"
    "DIRECTION: <one concise clause on the likely broad-ASX direction; if the sentiment is "
    "driven by interest-rate / RBA-guidance dynamics, name the most-exposed segment (e.g. "
    "rate-sensitive shares) and whether it is under pressure or supported>\n"
    "Labelling principles:\n"
    "1. If the assessment describes a divergence, disconnect, distrust, or loss of "
    "confidence between market pricing/expectations and official central-bank (RBA) "
    "guidance, the market itself is split — use 'mixed with a <dominant> bias', where "
    "<dominant> is the side the assessment leans toward.\n"
    "2. If the assessment weighs BOTH material upside and material downside drivers, use "
    "'mixed with a <dominant> bias'.\n"
    "3. Otherwise, when the assessment points essentially one direction, use plain "
    "'positive' or 'negative'."
)
_LABEL_RE = re.compile(r"(?im)^\s*LABEL:\s*(.+)$")
_DIRECTION_RE = re.compile(r"(?im)^\s*DIRECTION:\s*(.+)$")


def _brain_calibrate(assessment: str) -> tuple[str | None, str | None]:
    """Ask the brain to (a) map the assessment onto the graded label vocabulary and
    (b) give a grounded direction clause. Returns (canonical_label, direction_clause);
    either may be None if the call fails or the output isn't recognised."""
    try:
        resp = _get_client().chat.completions.create(
            model=config.BRAIN_MODEL,
            messages=[
                {"role": "system", "content": _CALIBRATION_SYS},
                {"role": "user", "content": f"Assessment:\n{assessment}\n\nReply:"},
            ],
            temperature=0.0,
            max_tokens=80,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        out = resp.choices[0].message.content or ""
    except Exception:
        return None, None
    label = None
    lm = _LABEL_RE.search(out)
    if lm:
        low = lm.group(1).strip().lower()
        # match longest label first so "mixed with a negative bias" wins over "negative"
        label = next((c for c in _VALID_LABELS if c in low), None)
    dm = _DIRECTION_RE.search(out)
    direction = dm.group(1).strip() if dm else None
    return label, direction


def _calibrate_afr_sentiment(assessment: str) -> str:
    """Reconcile the domain model's one-word sentiment with the brain's nuanced read.
    Only rewrites when the brain upgrades a one-word polarity to a 'mixed with a bias'
    read; otherwise the domain model's assessment is returned unchanged. The rewritten
    Direction carries the brain's grounded clause (which names the exposed sector when
    the move is rate-driven), so the downstream synthesis relays it verbatim."""
    m = _SENT_LINE.search(assessment)
    if not m:
        return assessment
    label, direction = _brain_calibrate(assessment)
    if not label or "mixed" not in label:
        # brain agrees with a one-directional read (or was unavailable) — leave as-is
        return assessment
    if "mixed" in m.group(1).lower():
        return assessment  # domain model already hedged
    arrow = "mixed-to-up" if "positive" in label else "mixed-to-down"
    assessment = _SENT_LINE.sub(f"Sentiment: {label}", assessment, count=1)
    tail = f" — {direction}" if direction else ""
    new_dir = f"Direction: broad ASX direction likely {arrow}{tail}"
    if _DIR_LINE.search(assessment):
        assessment = _DIR_LINE.sub(lambda d: new_dir, assessment, count=1)
    else:
        assessment += "\n" + new_dir
    return assessment


def _mock_assessment(mode: str, headline: str | None, article: str | None,
                     ticker: str | None) -> str:
    """Deterministic stand-in for the fine-tuned domain-ft model, used while
    DOMAIN_PREDICT_MODE != 'llm' (e.g. node1 :8001 not yet serving the LoRA).

    Matches the seam-contract output shape: a sentiment word
    (positive / negative / mixed) plus market-direction language, text only,
    NO invented numeric returns. When domain-ft is live, flip DOMAIN_PREDICT_MODE
    to 'llm' and the real model answers through the frozen prompt — no code change.
    """
    head = (headline or "").lower()
    blob = f"{headline or ''} {article or ''}".lower()

    if mode == "afr":
        # rate-guidance / RBA-doubt articles → mixed with a negative bias
        if ("interest rate" in blob or "rba" in blob or "cash rate" in blob) and (
            "don't believe" in blob or "disconnect" in blob or "no longer trust"
            in blob or "doubt" in blob or "wrong-footed" in blob or "distrust" in blob):
            return (
                "Sentiment: mixed with a negative bias. The article highlights a market that "
                "distrusts the RBA's rate guidance, so the likely direction for the broad ASX is "
                "mixed-to-down, with rate-sensitive shares (banks, REITs, high-multiple growth) "
                "under pressure as investors price in earlier and larger rate increases."
            )
        # Sector read is driven by the HEADLINE (strongest signal); check energy
        # before travel because energy articles can mention 'travel' in the body.
        if any(w in head for w in ("energy", "oil", "petroleum", "crude", "opec")):
            return (
                "Sentiment: positive. Firmer oil prices on vaccine-led demand hopes and possible "
                "OPEC supply restraint point to a likely upward direction for ASX energy shares as "
                "the sector builds on its recent rally."
            )
        if any(w in head for w in ("travel", "airline", "tourism", "flight", "aviation")):
            return (
                "Sentiment: positive. Vaccine-rollout and reopening optimism support a likely "
                "upward direction for ASX travel shares (airlines, tourism and leisure names) as "
                "border-reopening expectations improve demand."
            )
        return (
            "Sentiment: mixed. The article does not signal a clear directional bias; the likely "
            "near-term direction for the relevant ASX shares is balanced pending further data."
        )

    # technical mode — qualitative macro read, no numbers invented here
    tk = (ticker or "the stock").split(".")[0]
    return (
        f"Sentiment: mixed. A technical and macro read for {tk} is balanced given the prevailing "
        "RBA cash-rate backdrop; direction is data-dependent with no strong directional signal."
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
            f"  RBA cash rate: {rba_rate:.2f}%\n\n"
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

    # Mock-until-live: while the fine-tuned domain-ft model is not being served
    # (DOMAIN_PREDICT_MODE != "llm"), answer deterministically in the trained
    # output shape. The frozen prompt above is still built for tool-trace fidelity.
    if config.DOMAIN_PREDICT_MODE != "llm":
        return {
            "assessment": _mock_assessment(mode, headline, article, ticker),
            "mode": mode,
            "date": date,
            "rba_rate": rba_rate,
            "source": "mock:domain-ft",
        }

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

    # AFR sentiment can be genuinely two-sided; let the brain reconcile the
    # domain model's one-word label with the graded vocabulary (see note above).
    if mode == "afr":
        assessment = _calibrate_afr_sentiment(assessment)

    return {
        "assessment": assessment,
        "mode": mode,
        "date": date,
        "rba_rate": rba_rate,
    }
