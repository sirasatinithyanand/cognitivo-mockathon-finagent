"""LangGraph agent core — explicit StateGraph (Stage 2).

Flow: reason (LLM w/ tools bound) --tool_calls?--> act (execute via registry) --> reason
      reason --no tool calls--> synthesize (clean final answer) --> END

`retrieve` is just another registered tool, so retrieval and skill calls share one path.
"""
from __future__ import annotations

import json
import re
import sys
import time

from langgraph.graph import END, StateGraph

from tools import registry

from . import config, llm
from .state import AgentState

SYSTEM_PROMPT = """You are a financial research agent with access to three Australian financial datasets (RBA cash rate decisions 2010-2026, ASX stock prices 2015-2021 for 18 companies, and AFR news articles 2015-2021). Always call tools to get exact numbers — never invent figures.

TOOLS:

query_data(dataset="rba", metric=..., date_from=..., date_to=..., year=...)
  Metrics and what they return:
  - count              → total decision records in range
  - count_changes      → {result, increases, decreases, meta.total} — total records that changed the rate
  - count_increases    → {result, total_change_pp, rate_before_first_hike, by_year, first_date, last_date} — records where rate rose
  - count_decreases    → {result, total_change_pp, rate_before_first_cut, rate_after_last_cut, by_year, first_date, last_date}
  - count_holds        → {result, total, pct_of_total, no_change_years, by_year} — records where rate did NOT change; no_change_years lists years where EVERY decision was a hold
  - change_distribution → {distribution, most_common, total_non_hold} — frequency of each non-zero change size (e.g. how many -0.25pp cuts)
  - extremes           → {min_rate, min_first_date, min_record_count, max_rate, max_first_date, max_record_count} — highest and lowest cash-rate targets ever recorded
  - max_hold_streak    → {days, from_date, from_rate, to_date, to_rate} — longest gap between consecutive non-zero changes
  - min_change_interval → {days, from_date, to_date} — shortest gap between consecutive non-zero changes
  - lookup_rate        → rate IN EFFECT on or before a given date (use date_from for the target date)
  - list               → all records in range with date, change, rate
  - post_cut_basket_returns → per cut: {cut_date, new_rate, end_date, basket_return_pct} for 7-day post-cut ASX basket. ALWAYS pass exclude_tickers=["TAH.AX"].

query_data(dataset="asx", metric=..., ticker=..., year=..., date_from=..., date_to=..., exclude_tickers=...)
  Metrics:
  - lookup_price          → price on or nearest to date
  - annual_return         → first-to-last return for ticker in a single calendar year
  - rank_annual_returns   → {best, worst, top5, bottom3, all_tickers, avg_return_pct} sorted by return for a year
  - full_sample_return    → {best, worst, all_tickers, avg_return_pct, median_return_pct, positive_count, negative_count} first-to-last return across ENTIRE dataset (2015-2021). Use this for "full-sample", "total return", "overall return", "how many positive/negative", or "median return" questions.
  - price_return_between  → {per_ticker, basket_avg_return_pct} for a specific date window
  - volatility            → {highest3, lowest3, all_tickers} annualised volatility (std dev of daily log returns × √252)
  - max_volume            → {best, top5, all_tickers} single largest daily volume per ticker
  - avg_close / avg_volume → summary stats
  - max_drawdown → {worst3, worst, best3, best, all_tickers} — worst=largest drawdown, best3=3 smallest (least negative) losses, best3[0] is the absolute best
  - correlation → {correlations: [{pair, correlation, n_days}]} — pairwise Pearson correlation of daily log returns between tickers
  - cross_year_extremes → {worst, best, bottom5, top5} — best/worst annual return across ALL ticker-year combos 2015-2021
  - dataset_info          → tickers, date range, row count

query_data(dataset="afr", metric=..., pattern=..., year=..., date_from=..., date_to=...)
  All AFR metrics REQUIRE pattern= (a Python regex string, case-insensitive by default).
  Metrics:
  - count         → total records matching pattern in range
  - count_by_month → {records, best_month, best_year, total_matching} — top months by match count
  - share         → {result (%), matching, total, by_year} — fraction of records matching

retrieve(query=...)
  Semantic search over AFR articles. Returns ranked documents with headline and text.
  Use to find relevant articles before calling sentiment_assess.

sentiment_assess(mode=..., date=..., ...)
  Calls the fine-tuned financial domain model. Two modes:
  - mode="afr":       pass headline= and article= from a retrieved AFR article.
                      Returns a structured market sentiment + ASX impact assessment.
  - mode="technical": pass ticker= (e.g. "BHP.AX") and date=.
                      Returns a technical + macro assessment using daily price data.
  Use this for any qualitative question: "what was market sentiment", "how did the
  market react", "what was the technical picture for X". The model was trained on
  Australian financial news in context of the RBA cash rate — output is text, not a number.

domain_predict(ticker=..., horizon_days=...)
  Forward return forecast (numeric). Use only if explicitly asked for a price prediction.

REASONING RULES:
1. Always filter by date when the question names a period — use year=N or date_from/date_to.
2. count_decreases already returns total_change_pp and rate_before/after — no need to call sum_change separately.
3. count_changes returns the combined total (increases + decreases + meta.total for all records).
4. For multi-dataset questions, make one tool call per dataset.
5. For "one-week post-cut ASX returns": use post_cut_basket_returns (basket) + price_return_between per named ticker.
6. For "average annual return of all stocks": rank_annual_returns returns avg_return_pct directly.
7. For "share of articles matching X": always include pattern= in the call; by_year gives per-year breakdown.
8. Tabcorp (TAH.AX) is excluded from RETURN and VOLATILITY calculations only — always pass exclude_tickers=["TAH.AX"] to post_cut_basket_returns, rank_annual_returns, price_return_between, full_sample_return, and volatility. Do NOT exclude TAH.AX from avg_volume, max_volume, max_drawdown, or count metrics — include it when the question asks about all tickers or asks which ticker has the highest volume.
9. For "full-sample", "total return", "overall return", or "2015-2021 return": use full_sample_return NOT annual_return.
10. For "how many holds" or "hold records": use count_holds (returns no_change_years listing years with ZERO changes). For "most common change size": use change_distribution.
11. For "longest stretch between changes" or "longest hold period": use max_hold_streak. For "shortest interval between changes": use min_change_interval.
12. For "annualised volatility": use volatility metric. For "largest single-day volume": use max_volume.
13. For "first and last RBA record" or "net change between first and last": use dataset_info (returns first_record and last_record with date, change, rate).
14. For "years with no rate change" or "complete years with only holds": use count_holds — the no_change_years field lists them directly.
15. For "pairwise correlation" or "daily-return correlation": use correlation metric. Do NOT pass ticker= (it always loads all tickers). Use exclude_tickers=["TAH.AX"] as usual for returns.
16. For "worst single-year return" or "best single-year return" across all tickers and all years: use cross_year_extremes (covers all 2015-2021 ticker-year combinations in one call).
17. For "how many positive vs negative full-sample returns" or "median return": use full_sample_return — it returns positive_count, negative_count, median_return_pct directly.
18. For "best drawdowns" (smallest loss / least negative): use max_drawdown — best3[0] is the ticker with the smallest peak-to-trough loss. Do NOT exclude TAH.AX when the question asks about all 18 tickers.
19. For year-by-year comparison between two tickers (e.g. "which led each year 2015-2021"): call rank_annual_returns for EACH year separately (7 calls) and compare the two tickers.
20. AFR pattern rule: ALWAYS wrap short acronyms in \b word boundaries: use \bNAB\b not NAB, \bCBA\b not CBA, \bANZ\b not ANZ, \bQBE\b not QBE, \bAGL\b not AGL. Without \b, "NAB" matches words like "unable" and inflates counts by 5-10x."""

SYNTHESIS_PROMPT = (
    "You now have all the data needed. "
    "Write a direct factual answer in 1-3 sentences using ALL key numbers from the tool results "
    "(counts, totals, percentages, rates, dates, tickers). "
    "Start immediately with the facts — no preamble, no reasoning, no 'The user is asking'."
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TOOL_CALL_XML_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL | re.IGNORECASE)
_FINAL_MARKERS = re.compile(
    r"(?:Final Answer|Final Polish|Final Response|Final Check|Answer):\s*\n?(.*?)$",
    re.DOTALL | re.IGNORECASE,
)
# Qwen3 sometimes structures output as "*Draft 1:* <sentence>\n*Check constraints:*\n..."
_DRAFT_RE = re.compile(r'\*[Dd]raft\s*\d+:\*\s*(.+?)(?=\n\s*\*\s*\*[Cc]heck|\Z)', re.DOTALL)
_SKIP_PREFIXES = (
    "one ", "let ", "wait", "check", "i ", "i'", "the user", "looking",
    "draft:", "final check", "actually", "okay", "ok,", "i am", "i will",
    "so, ", "now,", "let's", "lets ", "here is", "here's", "based on",
    "this is", "these are", "this implies", "note:", "note,",
)


def _strip_think(text: str) -> str:
    """Extract clean factual answer from Qwen3 verbose output.

    Strategy (in order):
    1. Remove complete <think>...</think> blocks → return non-think remainder.
    2. Remove everything up to an orphan </think> → return remainder.
    3. Look for 'Final Answer:' / 'Final Polish:' markers → extract that section.
    4. Take the last paragraph that doesn't look like internal reasoning.
    """
    # Pre-clean: strip XML tool_call blocks that Qwen3 emits even without bound tools
    text = _TOOL_CALL_XML_RE.sub("", text)
    original = text.strip()
    text = original

    # Step 1: remove complete <think>...</think> blocks
    stripped = _THINK_RE.sub("", original).strip()
    if stripped != original.strip():
        return stripped if stripped else original.strip()

    # Step 2: orphan </think> — strip everything up to and including it
    if "</think>" in original:
        after = re.sub(r"^.*?</think>", "", original, flags=re.DOTALL).strip()
        if after:
            return after

    # Step 2.5: look for "*Draft N:* <sentence>" pattern (Qwen3 structured reasoning output)
    dm = _DRAFT_RE.search(original)
    if dm:
        candidate = dm.group(1).strip()
        if len(candidate.split()) >= 5:
            return candidate

    # Step 3: look for explicit "Final Answer:" / "Final Polish:" markers
    m = _FINAL_MARKERS.search(original)
    if m:
        candidate = m.group(1).strip()
        first_para = candidate.split("\n\n")[0].strip()
        return first_para if first_para else candidate

    # Step 4: pick the best paragraph — prefers one containing numbers, ignores footnotes/citations
    paras = [p.strip() for p in original.split("\n\n") if p.strip()]
    candidates = []
    for para in paras:
        low = para.lower()
        if any(low.startswith(x) for x in _SKIP_PREFIXES):
            continue
        if para.startswith("*(") or para.startswith("*Source") or para.startswith("_("):
            continue  # skip footnote / citation blocks
        if len(para.split()) < 5:
            continue  # too short
        candidates.append(para)

    if candidates:
        # Prefer paragraphs that contain a digit (factual answers have numbers)
        with_numbers = [p for p in candidates if re.search(r"\d", p)]
        return (with_numbers or candidates)[-1]  # last one wins

    # Fallback: return the original
    return original.strip()


def reason(state: AgentState) -> AgentState:
    try:
        resp = llm.chat(state["messages"], tools=registry.to_openai_tools())
    except Exception as e:
        err = str(e)
        if "ContextWindowExceeded" in err or "context" in err.lower() and "length" in err.lower():
            # Trim oldest tool results to free up context, then retry without tools
            trimmed = [m for m in state["messages"] if m.get("role") != "tool"]
            try:
                resp = llm.chat(trimmed, tools=None)
            except Exception:
                state["answer"] = "Context limit reached — unable to complete reasoning."
                return state
        else:
            raise
    msg = resp.choices[0].message
    entry: dict = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        entry["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]
    state["messages"].append(entry)
    if not msg.tool_calls:
        state["answer"] = _strip_think(msg.content or "")
    return state


def act(state: AgentState) -> AgentState:
    calls = state["messages"][-1].get("tool_calls", [])
    # Build a cache of already-executed (name, args) → result for dedup.
    _cache: dict[str, dict] = {
        json.dumps({"n": t["name"], "a": t["args"]}, sort_keys=True): t
        for t in state.get("tool_trace", [])
    }
    for tc in calls:
        name = tc["function"]["name"]
        args = json.loads(tc["function"]["arguments"] or "{}")
        cache_key = json.dumps({"n": name, "a": args}, sort_keys=True)

        if cache_key in _cache:
            # Re-use the cached result — avoids re-execution but preserves data in messages.
            cached = _cache[cache_key]
            # Reconstruct result from summary (best-effort: parse JSON, else wrap as string)
            try:
                result: dict = json.loads(cached["summary"]) if cached["summary"].startswith("{") else {"cached_result": cached["summary"]}
            except Exception:
                result = {"cached_result": cached["summary"]}
        else:
            t0 = time.perf_counter()
            try:
                result = registry.call(name, args)
            except Exception as e:  # noqa: BLE001
                result = {"error": str(e)}
            ms = int((time.perf_counter() - t0) * 1000)
            trace = {"name": name, "args": args, "ms": ms, "summary": str(result)[:200]}
            state.setdefault("tool_trace", []).append(trace)
            _cache[cache_key] = trace
            if name == "retrieve" and isinstance(result, dict):
                state.setdefault("sources", []).extend(result.get("docs", []))

        state["messages"].append(
            {"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(result)})
    state["steps"] = state.get("steps", 0) + 1
    return state


_SYNTH_SYS = (
    "/no_think\n"
    "You are a financial data assistant. "
    "Output ONLY the final factual answer — a plain English sentence or two. "
    "No bullet points, no drafts, no reasoning, no 'Based on the data'."
)

def synthesize(state: AgentState) -> AgentState:
    """Produce a clean factual answer using a compact synthesis call over the tool results."""
    user_q = next((m["content"] for m in state["messages"] if m["role"] == "user"), "")
    tool_results = [m["content"] for m in state["messages"] if m["role"] == "tool"]

    # Always synthesize from tool results when available — the model's raw answer often
    # omits numbers that appear in the results (e.g. says "20 increases, 21 decreases"
    # but forgets to state the 41 total and 175 denominator).
    if tool_results:
        tool_summary = "\n".join(f"Tool result {i+1}: {r}" for i, r in enumerate(tool_results[-8:]))
        prompt = (
            f"/no_think\n"
            f"Question: {user_q}\n\n"
            f"Tool data:\n{tool_summary}\n\n"
            "Write a direct factual answer in 1-4 plain sentences using ALL exact numbers from the tool data. "
            "Include every key number (counts, totals, percentages, rates, dates, tickers). "
            "Use the exact values from tool data — do not recompute or reformat numbers. "
            "Start with the answer immediately."
        )
        try:
            resp = llm.get_client().chat.completions.create(
                model=config.BRAIN_MODEL,
                messages=[
                    {"role": "system", "content": _SYNTH_SYS},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=400,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            synth = resp.choices[0].message.content or ""
            clean = _strip_think(synth) or synth
            if len(clean.split()) >= 5:
                state["answer"] = clean
                return state
        except Exception:
            pass

    # Fallback: strip think from whatever the model produced inline
    raw = state.get("answer", "")
    state["answer"] = _strip_think(raw) or raw
    return state


def _route(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    steps = state.get("steps", 0)
    if last_msg.get("tool_calls") and steps < config.MAX_AGENT_STEPS:
        return "act"
    if steps > 0:
        return "synthesize"
    return END




def build_graph():
    g = StateGraph(AgentState)
    g.add_node("reason", reason)
    g.add_node("act", act)
    g.add_node("synthesize", synthesize)
    g.set_entry_point("reason")
    g.add_conditional_edges("reason", _route, {"act": "act", "synthesize": "synthesize", END: END})
    g.add_edge("act", "reason")
    g.add_edge("synthesize", END)
    return g.compile()


def run(question: str) -> AgentState:
    state: AgentState = {
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": question}],
        "sources": [], "tool_trace": [], "steps": 0, "answer": "",
    }
    return build_graph().invoke(state)


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "Given current interest rates, should I hold BHP?"
    final = run(q)
    print("\n--- tool trace ---")
    for t in final["tool_trace"]:
        print(f"  {t['name']}({t['args']}) [{t['ms']}ms] -> {t['summary']}")
    print("\n--- sources ---")
    for s in final["sources"]:
        print(f"  [{s['id']}] {s['source']} (score={s['score']})")
    print("\n--- answer ---\n" + final["answer"])
