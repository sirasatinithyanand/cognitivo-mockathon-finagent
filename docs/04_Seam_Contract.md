# 04 — The Seam Contract (fine-tuning ⇄ agent boundary) — FROZEN

This is the single interface between the fine-tuning track and the agent track. Freeze it,
and both sides build in parallel with no blocking. **Change it only by agreement of all three.**

## The interface

The fine-tuned model is reached **only** through the `sentiment_assess` tool, which calls the
`domain-ft` alias on the LiteLLM proxy. Nothing else in the agent talks to the fine-tuned model.

```
agent  ──sentiment_assess(mode, date, headline, article | ticker)──▶
        LiteLLM proxy :4000  ──alias "domain-ft"──▶  vLLM node1 :8001  (Nemotron-8B + LoRA)
```

| Contract item | Frozen value |
|---------------|--------------|
| Alias | `domain-ft` (OpenAI-compatible `chat/completions` via LiteLLM `:4000`) |
| Env var | `DOMAIN_FT_MODEL=domain-ft` |
| Caller | `tools/sentiment_assess.py` (owned by the agentic team) |
| Callee | Nemotron-8B + LoRA served on node1 (owned by the fine-tuning team) |
| Max tokens | ~250 |

## Exact prompt formats the model is trained against

**AFR sentiment mode** (`mode="afr"`) — the one the scored questions use:

```
Date: {YYYY-MM-DD}
RBA cash rate: {rate}%
AFR Headline: {headline}
Article: {article body, ~800 chars}

As an Australian financial analyst, assess the market sentiment and likely ASX impact.
```

**Technical mode** (`mode="technical"`) — OHLCV read:

```
ASX daily data for {TICKER} on {YYYY-MM-DD}:
  Close: ${close} | High: ${high} | Low: ${low}
  Volume: {volume} | Daily range: {pct}%
  RBA cash rate: {rate}%

Provide a technical and macro assessment for {TICKER-short}.
```

The RBA rate is looked up automatically by the tool (`query_data lookup_rate`) — the model
must NOT compute it.

## Expected output

- Free text, ≤ ~250 tokens.
- Contains a **sentiment word**: `positive` / `negative` / `mixed` (synonyms preserving
  meaning are accepted by the grader).
- Contains **market-direction language** (e.g. "likely upward for ASX travel shares",
  "mixed-to-down", "rate-sensitive shares under pressure").
- **Do NOT** emit invented numeric returns or price forecasts — text assessment only.
  (This is an explicit organizer rule.)

### The 3 questions this must satisfy

| Q | Article | Expected sentiment | Expected direction |
|---|---------|-------------------|--------------------|
| MHQ058 | "Travel stocks take off on vaccine rollout" (23 Feb 2021, rate 0.10%) | positive | ASX travel shares upward |
| MHQ067 | "Why investors don't believe the RBA on interest rates" (25 Nov 2021, rate 0.10%) | mixed w/ negative bias | broad ASX mixed-to-down; rate-sensitive under pressure |
| MHQ080 | "Energy stocks shine as vaccines fuel oil rally" (28 Nov 2020, rate 0.10%) | positive | ASX energy shares upward |

Train the LoRA so the model reliably produces these sentiment/direction reads in the AFR
prompt format above. (The RBA rate and any ASX return maths are computed by tools, not the
model — the model only needs to get sentiment + direction right.)

## Mock-until-live: how the agent track stays unblocked

| Env var | Offline (mock) | Live (integration) |
|---------|----------------|--------------------|
| `LITELLM_BASE_URL` | `http://localhost:9000/v1` | `http://10.0.1.10:4000/v1` |
| `DOMAIN_PREDICT_MODE` | `mock` | `llm` |
| `DOMAIN_FT_MODEL` | `domain-ft` (unchanged) | `domain-ft` (unchanged) |

The agentic team develops against `mocks/mock_llm.py` (`:9000`) or `DOMAIN_PREDICT_MODE=mock`.
When the model is live, flip the two vars — **no code change**.

> Note: the shipped `mocks/mock_llm.py` returns a generic BHP/`domain_predict` canned answer.
> The agentic team should extend it (or the fixtures) so mock sentiment output roughly matches
> the format above — enough to exercise MHQ058/067/080 end-to-end before the real model lands.

## Integration checklist (the single convergence point)

1. **Fine-tuning team:** adapter trained, exported, served on node1 `:8001`; validated in
   isolation against the 3 articles above.
2. **Both teams:** register `domain-ft` in `~/litellm/config.yaml`; confirm
   `curl http://10.0.1.10:4000/v1/models` lists it.
3. **Agentic team:** set `DOMAIN_PREDICT_MODE=llm`, `LITELLM_BASE_URL=http://10.0.1.10:4000/v1`.
4. **Together:** re-run MHQ058 / MHQ067 / MHQ080 end-to-end; confirm sentiment + direction +
   correct RBA rate (0.10%) appear in each `answer`.
5. **Fine-tuning team:** fill the `model` block of `submission.json`
   (`endpoint: http://10.0.1.11:8001/v1`, `model_name: domain-ft`) — include only if the
   endpoint is reachable for technical review.

**Do not** change the `sentiment_assess` prompt strings without telling all three of us — a
silent change here means the fine-tuned model is prompted differently than it was trained,
and the sentiment questions quietly regress.
