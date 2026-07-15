# P2 — Agent + Skills + MCP + UI scaffold

Runnable reference for Stages 2–4 (see [`../p2_tasks.md`](../p2_tasks.md)).
Runs fully **offline** against `mocks/` — flipping env vars to the real stack requires zero code change.

## Asset → stage map

| Asset | Stage | Notes |
|-------|-------|-------|
| `agent/` (LangGraph StateGraph) | Stage 2 | reason ⇄ act loop over LiteLLM `agent-brain` |
| `tools/` (registry + 4 skills) | Stage 2 | one JSON-schema definition per tool (contracts §2) |
| `rag/query_client.py` | Stage 2–3 | `query(text, k, filters) -> list[Doc]` (contracts §3) |
| `mcp_server/` (FastMCP) | Stage 3 | same registry exposed as MCP tools (stdio + SSE) |
| `ui/` (FastAPI SSE + chat panel) | Stage 4 | contracts §4 frames: `token` / `sources` / `done` |
| `mocks/` + `demo/` | all | deterministic offline harness for the whole loop |

## Run order (offline)

```bash
pip install -r requirements.txt        # run everything below from p2_agent/

python -m demo.run_demo --mock         # starts mock LLM, runs the demo query, prints trace
python -m mcp_server.server --list-tools

uvicorn ui.app:app --port 8080         # then open http://localhost:8080
# (offline UI needs the mock up: python -m mocks.mock_llm  in another terminal)
```

Expected demo output: ≥1 `domain_predict` call, ≥1 `retrieve` call, a cited answer, the tool trace.

## Mock-until-live (env-var flip)

| Env var | Offline default | Real stack (integ #1, Jul 5) |
|---------|-----------------|------------------------------|
| `LITELLM_BASE_URL` | `http://localhost:9000/v1` (mock) | `http://<litellm-host>:4000/v1` |
| `LITELLM_KEY` | `EMPTY` | per-environment key |
| `BRAIN_MODEL` | `agent-brain` | `agent-brain` (unchanged — alias) |
| `QDRANT_URL` | unset → fixture `query()` | `http://<qdrant-host>:6333` |
| `DOMAIN_PREDICT_MODE` | `mock` (deterministic canned value) | `llm` → LiteLLM `domain-ft` (P1's model) |

## Contracts honored (`../p3_infra_harness/contracts.md`)

- **§1** — all LLM calls via the LiteLLM proxy aliases; vLLM is never called directly.
- **§2** — every tool: `{name, description, input_schema, output_schema}`; `domain_predict` matches the spec example field-for-field.
- **§3** — `Doc = {id, text, score, source, metadata}`; same shape flows into the UI `sources[]` payload.
- **§4** — SSE frames `{"type":"token"}` ... `{"type":"sources"}` → `{"type":"done"}`.
