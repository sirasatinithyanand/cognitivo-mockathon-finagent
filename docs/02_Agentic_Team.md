# 02 — Agentic Team (2 people): the whole agent

You build everything except the fine-tuned model — the tools that produce exact numbers,
AFR retrieval, the LangGraph agent, the Qwen brain + LiteLLM serving, and the `/query`
endpoint the judges hit. Read `01_Environment_and_Gotchas.md` first.

**Split for 2 people (suggested):**
- **Workstream 1 — Tools & RAG & validation** → Steps 2–4 (`query_data`, `retrieve`, sentiment caller, checking all 15 questions). *This is the biggest share of the score.*
- **Workstream 2 — Agent loop, serving & endpoint** → Steps 5–8 (agent graph, Qwen brain, LiteLLM proxy, `server.py`, judge endpoint).

Steps 0–1 are shared setup. Do them together, then split.

---

## Step 0 — Setup (shared, 15 min)

```bash
SRC=~/Cognitivo_Training/AI_Training_and_Hackathon/Sample_Activity_5/p2_agent
cp -r "$SRC" ~/team01-agent && cd ~/team01-agent    # ONE shared working copy (git it early)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export DATA_DIR="/home/cognitivo/Desktop/HackathonDataset/Jsonl format DataSets"
export DOMAIN_PREDICT_MODE=mock
```

## Step 1 — Offline smoke (shared, 15 min)

```bash
python -m mocks.mock_llm &                     # mock brain on :9000
export LITELLM_BASE_URL="http://localhost:9000/v1"
python -m demo.run_demo --mock                 # expect a tool trace + a cited answer
```
If it errors on data → `DATA_DIR` wrong (Desktop, not Downloads) or RBA not read `utf-8-sig`.

---

## Workstream 1 — Tools, RAG & validation

### Step 2 — Prove the data loads
```bash
python - <<'PY'
from tools.query_data import query_data
print(query_data(dataset="rba", metric="dataset_info"))
print(query_data(dataset="asx", metric="dataset_info"))   # 18 tickers, 1774 rows, 2015-01-02..2021-12-30
print(query_data(dataset="afr", metric="count", pattern=r"\bunemployment\b", year=2020))
PY
```

### Step 3 — Validate every numeric question against `expected_fact` (core work)
`tools/query_data.py` already implements all metrics (~934 lines). For each line in
`public_questions.jsonl`, confirm the answer contains **every** `expected_fact`. Metric map:

| Q | Dataset | Metric |
|---|---------|--------|
| MHQ001 | RBA | `count_changes` → total / increases / decreases |
| MHQ035 | RBA | `count_decreases` (year) → cuts, total_change_pp, start/end target |
| MHQ040 | ASX | `dataset_info` → 18 tickers, 1774 rows, date range |
| MHQ045 | ASX | `rank_annual_returns(year=2018, exclude_tickers=["TAH.AX"])` |
| MHQ049 | ASX | `avg_volume` (do NOT exclude TAH) |
| MHQ055 | ASX | `max_drawdown` → worst3 + peak/trough dates |
| MHQ061 | AFR | `count_by_month(pattern=r"\bunemployment\b")` |
| MHQ076 | AFR+ASX | AFR `count(pattern=r"\bQBE\b", year=2021)` + ASX `annual_return` + rank |
| MHQ072/074 | RBA+ASX | `post_cut_basket_returns` + `price_return_between` per ticker |
| MHQ084 | RBA+AFR+ASX | `count_decreases` + AFR `count` + `rank_annual_returns` avg |
| MHQ090 | all | coverage check — AFR/ASX end 2021 → answer is **"No, not supported"** |
| MHQ058/067/080 | AFR(+RBA/ASX) | **sentiment** → `retrieve` + `sentiment_assess` (Step 4) |

One-shot check harness:
```bash
python - <<'PY'
import json
from agent.graph import run
P="/home/cognitivo/Cognitivo_Training/Mock_Hackathon_Participant_Package/public_questions.jsonl"
for line in open(P):
    q=json.loads(line); out=run(q["prompt"])
    print(q["id"],"=>",out["answer"][:200])
    print("   expects:",[c["expected_fact"] for c in q["grading"]["components"]])
PY
```
Bar: every `expected_fact`'s key numbers/dates appear in the answer. When off, fix the
**metric/args** (not the LLM prompt). Usual causes: missing `exclude_tickers`, wrong AFR
scope, missing `\b`, wrong date window.

### Step 4 — AFR retrieval + `sentiment_assess` caller (on mock)
- `rag/query_client.py` falls back to keyword-overlap over `mocks/fixtures/docs.json` when
  `QDRANT_URL` is unset. **Populate that fixture** with the 3 sentiment-question articles
  (pull real bodies from the AFR JSONL by headline/date):
  - "Travel stocks take off on vaccine rollout" (23 Feb 2021)
  - "Why investors don't believe the RBA on interest rates" (25 Nov 2021)
  - "Energy stocks shine as vaccines fuel oil rally" (28 Nov 2020)
  Doc shape: `{"id","text","source","metadata":{"date","headline"}}`.
- Keep `sentiment_assess` prompt format EXACTLY as in `04_Seam_Contract.md` (that's what the
  fine-tuned model expects). Extend `mocks/mock_llm.py` so its canned reply looks like a real
  sentiment read (sentiment word + direction) — enough to exercise MHQ058/067/080 before the
  real model lands.

---

## Workstream 2 — Agent loop, serving & endpoint

### Step 5 — Fix config + serve `/query`
1. `agent/config.py`: set the `DATA_DIR` default to the Desktop path (or rely on env).
2. Serve and hit it exactly like the judges:
```bash
python server.py            # :5000 (PORT to override)
curl localhost:5000/health
curl -s localhost:5000/query -H 'content-type: application/json' \
  -d '{"question":"From the first RBA record to the last, how many cash-rate decisions changed the rate, and how many were increases versus decreases?"}' | python -m json.tool
```
Response must match `answer.json` shape and pass `validate.json` (`answer` non-empty string).

### Step 6 — Bring up Qwen brain + LiteLLM proxy (node0) — DO THIS EARLY
Shared spine; the fine-tuning team registers `domain-ft` into the same proxy.
```bash
# node0 (10.0.1.10) — use `sg docker -c '...'` (docker-group quirk)
bash ~/cluster-scripts/bootstrap_cluster.sh          # starts vLLM brain (Qwen3.6-35B :8000), ~4 min
sg docker -c 'docker run -d --name litellm-proxy --network host \
  -v ~/litellm/config.yaml:/app/config.yaml ghcr.io/berriai/litellm:main-latest \
  --config /app/config.yaml --port 4000'
```
`~/litellm/config.yaml` — define BOTH aliases (fine-tuning team fills `domain-ft`'s model):
```yaml
model_list:
  - model_name: agent-brain
    litellm_params: { model: openai/Qwen3.6-35b-A3B-FP8, api_base: http://10.0.1.10:8000/v1, api_key: EMPTY }
  - model_name: domain-ft
    litellm_params: { model: openai/domain-ft,          api_base: http://10.0.1.11:8001/v1, api_key: EMPTY }
```
Then point the agent at the real proxy:
```bash
export LITELLM_BASE_URL="http://10.0.1.10:4000/v1"; export BRAIN_MODEL="agent-brain"
```

### Step 7 — Tune the loop for scoring
- `MAX_AGENT_STEPS` (default 6) — enough for 3-dataset questions (MHQ080/084); bump if truncating.
- Qwen verbosity: `agent/graph._strip_think()` removes `<think>`/draft scaffolding; synthesis
  uses `enable_thinking: False`. Extend strip rules if reasoning leaks into `answer`.
- Numbers must survive synthesis — the synth prompt says "use exact tool values, do not
  recompute". Keep that; the LLM never re-derives numbers.

### Step 8 — Expose to judges + `submission.json` (agent block)
- Stable URL (model box public address or a tunnel); `/health` must 200.
```json
"agent": { "endpoint": "https://<host>", "health_path": "/health",
           "query_path": "/query", "timeout_seconds": 300 }
```

---

## Definition of done (agentic team)
- [ ] All 12 numeric questions hit every `expected_fact`; AFR counts use combined-field, case-insensitive, once-per-record, `\b`-anchored search; Tabcorp excluded from returns/vol only.
- [ ] `retrieve` returns the 3 sentiment articles; `sentiment_assess` works on the mock.
- [ ] `server.py` serves `/query` + `/health`; runs on the **real Qwen brain**, not just the mock.
- [ ] LiteLLM proxy up on `:4000` with `agent-brain` + a `domain-ft` slot the fine-tuning team can fill.
- [ ] All 15 questions complete within `timeout_seconds`; endpoint reachable off-box; `agent` block filled.
- [ ] No endpoints/keys hard-coded.
