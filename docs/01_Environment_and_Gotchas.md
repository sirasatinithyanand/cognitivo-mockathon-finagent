# 01 — Environment, Paths & Known Gotchas (everyone read)

## Datasets (on this machine — NOTE the path)

```
DATA_DIR="/home/cognitivo/Desktop/HackathonDataset/Jsonl format DataSets"
```

⚠️ **The scaffold's `agent/config.py` hard-codes the WRONG default**
(`~/Downloads/Jasonl format DataSets`). The data is actually on the **Desktop**.
Export `DATA_DIR` (see below) or the loaders return nothing.

| Dataset | Folder | Files | Fields |
|---------|--------|-------|--------|
| RBA cash rate | `RBA-Rates-2010-2026/RBA-rates.jsonl` | 1 file | `Effective Date`, `Change % points`, `Cash rate target%` |
| ASX prices | `ASX-18-companies-2015-2021-Jasonl/*.jsonl` | 18 files | `ticker, date, open, high, low, close, volume` |
| AFR news | `AFR Jasonl/*.jsonl` | 86 monthly files | `HEADLINE, SUBHEAD, INTRO, TEXT, NEWSPAPER, PUBLICATIONDATE` |

- **ASX:** 18 tickers, **1,774 rows each**, date range **2015-01-02 → 2021-12-30**.
- **RBA:** decisions 2010–2026. File is **UTF-8 with a BOM** → open with `encoding="utf-8-sig"`.
  `Change % points` is a string like `"+0.25"`, `"-0.25"`, `"0.00"`.
- **AFR:** coverage **ends in 2021** (matters for MHQ090 — the 2022-23 "not supported" answer).

## Models (already on disk)

```
/home/cognitivo/local-llm-setup/models/Llama-3.1-Nemotron-Nano-8B-v1   <- fine-tune target (domain-ft)
/home/cognitivo/local-llm-setup/models/Qwen3.6-35b-A3B-FP8             <- brain (agent-brain)
```

## Cluster (2× DGX Spark GB10)

| Service | Node | Port | LiteLLM alias | Owner |
|---------|------|------|---------------|-------|
| vLLM brain — Qwen3.6-35B | node0 `10.0.1.10` | `:8000` | `agent-brain` | Agentic team |
| LiteLLM proxy | node0 `10.0.1.10` | `:4000` | (routes both aliases) | Agentic team |
| vLLM domain-ft — Nemotron-8B + LoRA | node1 `10.0.1.11` | `:8001` | `domain-ft` | Fine-tuning team |

- LiteLLM config lives at `~/litellm/config.yaml`. Run the proxy with `--network host` so it
  can reach `10.0.1.10:8000` and `10.0.1.11:8001`.
- Cluster bootstrap: `~/cluster-scripts/bootstrap_cluster.sh`.
- **Agent code calls only the LiteLLM proxy (`:4000`)** — never vLLM hosts directly.

## Environment variables (the only place endpoints live — never hard-code)

```bash
# --- shared ---
export DATA_DIR="/home/cognitivo/Desktop/HackathonDataset/Jsonl format DataSets"
export LITELLM_BASE_URL="http://10.0.1.10:4000/v1"   # offline mock: http://localhost:9000/v1
export LITELLM_KEY="EMPTY"                            # real key injected by harness at scoring
export BRAIN_MODEL="agent-brain"
export DOMAIN_FT_MODEL="domain-ft"
export MAX_AGENT_STEPS=6

# --- until the fine-tuned model is live, stay on the mock ---
export DOMAIN_PREDICT_MODE="mock"                     # flip to "llm" at integration

# --- RAG: unset QDRANT_URL => offline keyword-overlap fixture fallback ---
# export QDRANT_URL="http://<host>:6333"
# export QDRANT_COLLECTION="afr"
# export EMBED_MODEL="embed"
```

## Known gotchas (these have bitten before)

1. **Dataset path** — Desktop, not Downloads (above). #1 cause of "empty results".
2. **RBA BOM** — must read with `utf-8-sig` or the first key becomes `﻿Effective Date`.
3. **AFR search scope is graded and non-negotiable** — counts MUST search
   `HEADLINE + SUBHEAD + INTRO + TEXT` **combined**, case-insensitive, **once per record**.
   Whole-word acronyms MUST use `\b` anchors (`\bNAB\b`, not `NAB` — otherwise "unable"
   etc. inflates counts 5-10×). Wrong scope → counts won't match the reference answers.
4. **Tabcorp (`TAH.AX`)** — excluded from **returns & volatility only** (pass
   `exclude_tickers=["TAH.AX"]`). **Include** it for volume / drawdown / count questions.
5. **Docker access quirk** — `cognitivo` is in the `docker` group but a pre-existing shell
   doesn't show it. Use `sg docker -c 'docker ...'` (no password) instead of `sudo`.
6. **Fine-tune / GB10** — use container `nvcr.io/nvidia/nemo:25.09+` (25.04 crashes on the
   first NCCL kernel). Always train inside `tmux` (earlyoom prefer-kills `python3`).
7. **No trained adapter exists yet** — the `domain-ft` endpoint can't serve until the
   fine-tuning team produces one. That's exactly why the agentic team runs on the mock.
8. **Never hard-code endpoints/keys** in source or logs — the harness injects `LITELLM_KEY`
   and base URLs at scoring time.

## Reference material on disk

```
~/Cognitivo_Training/Mock_Hackathon_Participant_Package/
  public_questions.jsonl   # 15 calibration questions + expected_fact grading
  Challenge_Brief.html     # full rules, scoring, tech reference
  submission-guide.md      # repo layout + /query API contract
  question.json answer.json validate.json   # exact request/response schema
```
