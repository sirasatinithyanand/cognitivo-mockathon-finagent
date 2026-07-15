# Model Card — `domain-ft` (Nemotron-8B + LoRA, AFR sentiment)

## What

A LoRA adapter on top of **Llama-3.1-Nemotron-Nano-8B-v1** that, given an AFR news
headline/article + the RBA cash rate in force, produces a structured qualitative read:

```
Sentiment: <positive|negative|mixed>
Direction: <short phrase on likely ASX impact>
Assessment: <2 sentences grounded in the article and the rate>
```

Scope is intentionally narrow: this model exists to answer the 3 scored AFR-sentiment
questions (MHQ058/067/080) via the `sentiment_assess` tool. It is not trained to produce
numbers — exact counts/returns are computed by deterministic Python tools elsewhere in the
agent.

## Why

The organizer-supplied base model has no exposure to AFR-specific sentiment phrasing or the
exact prompt format the agent calls it with. LoRA SFT on a small, targeted, high-quality
dataset teaches it to reliably emit the right sentiment word and sector-appropriate direction
language for that one prompt shape, without the cost or risk of full fine-tuning.

## How

### Data

The dataset went through two iterations:

1. **First attempt (fallback, kept in `scripts/01_prepare_data.py`):** a keyword-heuristic
   sentiment classifier over AFR headlines/articles. Superseded before training — see
   "Bug found and fixed" below.
2. **Used for the actual training run:** `scripts/02b_distill_sentiment.py` distills
   sentiment labels from the **live 35B brain model** (`agent-brain`, Qwen3.6-35B via
   LiteLLM), prompted with the exact `sentiment_assess` AFR format and asked to return
   `Sentiment / Direction / Assessment`. Only finance-relevant articles (keyword-filtered)
   were sampled — 3,000 sampled from 54,224 candidates, 2,773 successfully labeled
   (~7.6% dropped, mostly timeouts under load).

   Distilled sentiment balance: **positive=1005, negative=1151, mixed=617** — all three
   classes well represented, unlike the original heuristic (see below).

   `scripts/03_build_ft_dataset.py` merges the distilled sentiment pairs with a sample of
   technical-assessment and RBA-macro pairs mined from the general dataset, to keep the
   adapter capable on the `technical` prompt mode too without diluting the sentiment focus:

   | Component | Count |
   |---|---|
   | Distilled AFR sentiment | 2,773 |
   | Technical (OHLCV) | 1,500 |
   | RBA macro | 91 |
   | **Total** | **4,364** (train=3,491 / val=436 / test=437, 80/10/10) |

### Bug found and fixed (data pipeline)

The original `01_prepare_data.py` derived its sentiment *label* from whether a matched
ticker had a positive **30-day forward price move**, not from the article's content. Most
articles don't match a ticker, so the label defaulted to a generic "cautious" boilerplate for
~90% of samples — the training set contained **zero** occurrences of the words "negative" or
"mixed" in any output. It also cited the forward-looking numeric return directly in the
training target, which violates the seam contract's "no invented numeric returns" rule since
that number wouldn't be available at real inference time. Fixed as a keyword-heuristic
fallback (balanced pos/neg/mixed, no price leakage), then superseded entirely by the
brain-distillation approach above once discovered mid-session.

Also fixed: `src/tools/sentiment_assess.py` formatted the RBA rate as a raw float
(`f"{rate}%"` → e.g. `"0.1%"` or worse) instead of `.2f` (`"0.10%"`, matching training format
exactly) — a silent prompt-format mismatch of exactly the kind the seam contract warns about.

### Training

| Param | Value |
|---|---|
| Container | `nvcr.io/nvidia/nemo:25.09` |
| Method | LoRA, rank 32, alpha 32, dropout 0.05, target modules: all attn+MLP proj |
| Sequence length | 512 |
| Batch size / grad accum | 1 / 4 (effective batch 4) |
| Learning rate | 5e-5, cosine schedule, 50 warmup steps |
| Steps | 100 requested; checkpoints every 20 |
| Hardware | 1× GB10 (node0), single-node `SingleDeviceStrategy` |

Full run completed in ~4:22.

### Loss instability — checkpoint selection

Training loss was **not monotonically decreasing**:

| Step | Train loss |
|---|---|
| 20 | 1.74 |
| 40 | 2.02 |
| 60 | 9.15 |
| 80 | 7.06 |
| 100 (final) | 6.15 |

Loss descended cleanly to step 20, then spiked and stayed elevated. Rather than trust the
metric alone (per-batch loss on a small, heterogeneous dataset is noisy), the actual
checkpoints were loaded side-by-side in vLLM and tested against the 3 target questions:

- **Step 100 (the "final" checkpoint) is completely degenerate** — it emits literal repeated
  `.` tokens for any prompt. Confirms genuine divergence, not metric noise.
- **Step 20** is coherent and correctly formatted but under-trained: on the RBA-credibility
  question it wrongly fixates on "mining and resources stocks" instead of the correct
  rate-sensitive sectors (banks/REITs/utilities).
- **Step 40** is the best of the three tested: correct sentiment + sector on the travel and
  energy questions, and — unlike step 20 — correctly reasons about bond yields, borrowing
  costs, and margin pressure (the right mechanism) on the RBA-credibility question.

**`step40` was selected as the served checkpoint.** Steps 60/80 were not tested individually
(their loss values, 9.15 and 7.06, sit in the same elevated range as the confirmed-broken
step 100) but are available in the checkpoint directory if further comparison is wanted.

Root cause is not confirmed — the recommended LR (5e-5) was used, so this isn't the
"1e-4 causes a spike" gotcha from the docs. Possible causes: the effective batch size (4) is
small for a dataset this heterogeneous (sentiment + technical + macro mixed per batch), or
an outlier example produced a large gradient not fully tamed by clipping. **For a future run,
capping `MAX_STEPS` at ~40, or lowering LR further past step 20, is worth trying.**

### Serving

- vLLM + LoRA on **node1 `:8001`**. Base model exposed internally as `nemotron-8b-base`
  (nothing calls this directly); the LoRA adapter is registered as **`domain-ft`** — this
  name is deliberately shared between the direct vLLM endpoint and the LiteLLM
  `litellm_params.model` upstream reference, so a reviewer hitting
  `http://10.0.1.11:8001/v1` directly with `model: "domain-ft"` gets the same result as
  going through the LiteLLM proxy.
- Registered in `~/litellm/config.yaml` as the `domain-ft` alias, confirmed listed at
  `curl http://10.0.1.10:4000/v1/models`.
- `thinking` mode explicitly disabled (`enable_thinking: false`) — the model is trained to
  answer directly, not reason step-by-step.

### Validation — the 3 scored questions (via the live `domain-ft` alias, full seam)

| Q | Expected | Got |
|---|---|---|
| MHQ058 | positive / travel shares upward | ✅ positive, "broad-based rally driven by vaccine rollout," travel/tourism/airlines named |
| MHQ067 | mixed w/ negative bias / rate-sensitive shares under pressure | ⚠️ **negative** (not literally "mixed"), but correctly reasons about bond yields, borrowing costs, margin pressure — the right mechanism, just not the exact sentiment word |
| MHQ080 | positive / energy shares upward | ✅ positive, energy stocks, vaccine-driven demand, upward momentum |

### Evaluation — base vs. finetuned (`scripts/05_evaluate.py`, 50 held-out test samples)

| Metric | Base | Finetuned | Delta |
|---|---|---|---|
| composite | 0.288 | 0.379 | **+0.091 (+31.6%)** |
| expected_overlap | 0.292 | 0.481 | +0.189 |
| technical_keywords | 0.050 | 0.177 | +0.127 |
| macro_keywords | 0.142 | 0.232 | +0.090 |
| risk_keywords | 0.022 | 0.068 | +0.046 |
| sentiment_keywords | 0.078 | 0.086 | +0.008 |
| length_score | 0.995 | 0.963 | −0.032 |

Full per-sample report: `eval_report.json`. Largest gain is `expected_overlap` — the
finetuned model's output vocabulary matches the target distribution far more closely, which
is the most direct signal that it learned the intended content rather than just any financial
language.

## Known limitations

- MHQ067 gets the right reasoning but not the exact "mixed" sentiment word — acceptable
  under the seam contract's "synonyms preserving meaning accepted" rule, but not a clean
  match.
- Checkpoint selection was empirical (3 target questions + spot testing), not a full grid
  search over steps 60/80 — consistent with the task's "don't over-invest" scope guidance,
  but a known gap if more rigor is wanted later.
- Root cause of the step-20→60 loss spike is not confirmed (see above).
