# Team-01 — Cognitivo Mock Hackathon — Build Runbook

**Read this first.** Two teams, one seam between them. Then go to your team's runbook.

| Doc | Who | What |
|-----|-----|------|
| this file | everyone | overview, teams, timeline |
| `01_Environment_and_Gotchas.md` | everyone | paths, cluster, dataset schemas, known traps |
| `02_Agentic_Team.md` | **Agentic team** | the whole agent: tools, RAG, agent loop, Qwen brain, LiteLLM, `/query` server |
| `03_FineTuning_Team.md` | **Fine-tuning team** | LoRA SFT of Nemotron-8B, serving the adapter, `domain-ft` alias, evidence |
| `04_Seam_Contract.md` | everyone | the frozen `domain-ft` interface — the only thing the two teams share |

---

## What we are building

An agent that answers Australian-financial questions over **three local datasets**
(RBA cash-rate decisions, ASX 18-stock prices, AFR news) and serves them as an HTTP
endpoint the judges call:

```
POST /query   {"question": "..."}   ->   {"answer": "...", "steps": N, "tool_trace": [...]}
GET  /health  -> 200
```

Scoring is **component-level factual correctness** (exact counts, returns, drawdowns,
dates, sentiment). Only the `answer` field is graded; `steps`/`tool_trace` feed the
leaderboard tool-usage column. Full rules in
`~/Cognitivo_Training/Mock_Hackathon_Participant_Package/` (Challenge Brief + `submission-guide.md`).

## Architecture

```
POST /query ─▶ LangGraph agent (server.py)
                │  brain = Qwen3.6-35B  ("agent-brain" via LiteLLM)  ← does tool-calling
      reason ⇄ act loop
                │
   ┌────────────┼───────────────────────────────┐
   ▼            ▼                                 ▼
query_data    retrieve (RAG over AFR)      sentiment_assess
(RBA/ASX/AFR   ranked articles            ─▶ domain-ft = Llama-Nemotron-8B + LoRA
 EXACT numbers,                               (fine-tuning team's model — sentiment only)
 pure Python)
                │
             synthesize ─▶ {answer, steps, tool_trace}
```

**Key principle:** all exact numbers come from `query_data` (Python), **never** from the
LLM. The fine-tuned Nemotron is used for **only the 3 AFR sentiment questions**
(MHQ058 / MHQ067 / MHQ080). Most of the score lives in `query_data` — get it right first.

## The two teams

| Team | Members | Owns | Runs on |
|------|---------|------|---------|
| **Fine-tuning** | You | `training/`, data prep, LoRA SFT, adapter serving, `domain-ft` alias, model card, `model.*` in `submission.json` | node1 `10.0.1.11:8001` |
| **Agentic** | The other 2 | `src/`: `query_data` tools, RAG `retrieve`, `sentiment_assess` caller, LangGraph agent, Qwen brain, **LiteLLM proxy** `:4000`, `server.py`, judge endpoint, `agent.*` in `submission.json` | node0 `10.0.1.10:8000` + `:4000` |

The agentic team is 2 people — divide the work internally however you like. A natural split:
one person on **tools + RAG + validation** (`query_data`, `retrieve`, checking all 15
questions), the other on **agent loop + serving + endpoint** (`graph.py`, Qwen brain,
LiteLLM, `server.py`). See `02_Agentic_Team.md` — it's structured as those two workstreams.

## The one rule that keeps both teams unblocked

The agentic team codes against the **mock domain-ft** (`DOMAIN_PREDICT_MODE=mock`). Nobody
waits on training. The fine-tuned model drops in later with a **single env-var flip** — no
code change. The frozen interface is in `04_Seam_Contract.md`. **Neither team changes the
`sentiment_assess` prompt format without telling the other** — that is the seam.

## Dependency order

1. **Agentic team brings up the LiteLLM proxy** (`:4000`) + Qwen brain first — it's the
   shared spine (routes `agent-brain` for them, `domain-ft` for the fine-tuning team). Until
   then, develop against the offline `mocks/mock_llm.py` on `:9000`.
2. **Agentic team** builds tools + validates the 15 public questions entirely on mocks.
3. **Fine-tuning team** trains + serves + validates the model in isolation.
4. **Converge once:** register `domain-ft` into the LiteLLM proxy → flip mock→live →
   re-run MHQ058/067/080.

## Base scaffold — copy, don't rewrite

The working reference agent already exists. Start from it:

```
SRC=~/Cognitivo_Training/AI_Training_and_Hackathon/Sample_Activity_5/p2_agent
```

Fine-tuning scaffold (scripts, configs, prep):

```
FT=~/Cognitivo_Training/finagent-finetune-participant
```

**Do NOT** build the agent on `~/Cognitivo_Training/finagent-finetune-participant/agent/` —
that's a yfinance live-quote demo, uses the wrong (internet) data, and is not scored.

## Definition of done (team)

- [ ] All 15 questions in `public_questions.jsonl` return a non-empty `answer` hitting the `expected_fact` components.
- [ ] `/query` + `/health` reachable at a stable URL for judges.
- [ ] Response validates against `validate.json`.
- [ ] Fine-tuned Nemotron used at inference for the sentiment questions.
- [ ] Repo layout: `README.md`, `submission.json`, `answer.json`, `src/`, `training/`, `logs/`.
- [ ] No credentials/endpoints hard-coded — everything via env vars.
