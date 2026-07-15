# Cognitivo Mock Hackathon — Team-01 Financial Agent (finagent)

An evidence-grounded market-signal agent that answers questions over three approved
Australian financial datasets (**RBA** cash-rate decisions, **ASX** 18-stock prices,
**AFR** news) and serves them at an HTTP endpoint the evaluator calls.

> Status: scaffold. `src/` starts from the reference agent; `training/` holds the
> fine-tuning pipeline. See `docs/` for the full team build runbooks.

## Architecture

```
POST /query -> LangGraph agent (src/server.py)
                |  brain = Qwen3.6-35B  ("agent-brain" via LiteLLM)  <- tool-calling / reasoning
      reason <-> act loop
                |
   +------------+-------------------------------+
   v            v                               v
query_data    retrieve (RAG over AFR)      sentiment_assess
(RBA/ASX/AFR   ranked articles            -> domain-ft = Llama-3.1-Nemotron-Nano-8B + LoRA
 EXACT numbers,                               (our fine-tuned model - sentiment questions only)
 pure Python)
                |
             synthesize -> {answer, steps, tool_trace}
```

- **All exact numbers** (counts, returns, drawdowns, dates) come from deterministic Python
  in `src/tools/query_data.py` — never invented by the LLM.
- **The fine-tuned Nemotron** (`domain-ft`) is used only for the AFR sentiment/direction
  questions, via `src/tools/sentiment_assess.py`.
- **All LLM traffic** routes through the LiteLLM proxy aliases (`agent-brain`, `domain-ft`) —
  vLLM hosts are never called directly.

## Run

```bash
cd src
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export DATA_DIR="/home/cognitivo/Desktop/HackathonDataset/Jsonl format DataSets"
export LITELLM_BASE_URL="http://10.0.1.10:4000/v1"   # offline mock: http://localhost:9000/v1
export BRAIN_MODEL="agent-brain"
export DOMAIN_FT_MODEL="domain-ft"
export DOMAIN_PREDICT_MODE="llm"                     # "mock" until the fine-tuned model is live

python server.py                                     # serves :5000
```

Offline demo (no cluster): `python -m mocks.mock_llm &` then `python -m demo.run_demo --mock`.

## API contract

```http
POST /query    {"question": "..."}
GET  /health
```

Response (see `answer.json`; only `answer` is graded):

```json
{ "answer": "...all requested components...", "steps": 3, "tool_trace": [ {"tool":"...","args":{},"result":"..."} ] }
```

## Fine-tuning summary

- **Base model:** Llama-3.1-Nemotron-Nano-8B-v1 (organizer-supplied).
- **Method:** LoRA SFT (NeMo `25.09`, rank 32, seq 512, LR 5e-5, ~100 steps) on GB10.
- **Data:** AFR news + ASX OHLCV + RBA rates, prepared into instruction pairs weighted
  toward the sentiment-assessment prompt format the agent calls.
- **Served:** vLLM + LoRA on node1 `:8001`, exposed as the LiteLLM `domain-ft` alias.
- Scripts, config, prep, logs, metrics, and model card live in `training/`.

## Repository layout

```
README.md          this file
submission.json    team + repo commit + agent/model endpoint registration (fill before deadline)
answer.json        sample response demonstrating the per-question contract
src/               agent source, tools, RAG, LiteLLM client, /query server
training/          fine-tuning scripts, configs, prep, logs, model card
logs/              non-sensitive run logs
docs/              team build runbooks (agentic team, fine-tuning team, seam contract)
```

## Known limitations

- AFR + ASX coverage ends in 2021; questions about 2022-2023 are answered as **not
  supported by the data** (stated honestly rather than fabricated).
- `domain-ft` returns qualitative sentiment/direction text, not numeric forecasts.
- AFR pattern counts use combined-field (`HEADLINE+SUBHEAD+INTRO+TEXT`), case-insensitive,
  once-per-record, word-boundary matching — required for reproducible counts.

## Datasets & secrets

Organizer-supplied datasets are **not** committed (see `.gitignore`). Endpoints and
credentials are read from environment variables only — never hard-coded.
