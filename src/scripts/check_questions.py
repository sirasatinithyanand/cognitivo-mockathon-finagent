#!/usr/bin/env python3
"""Shared validation harness — run all public questions through the agent and check
that every expected_fact's key numbers/dates land in the answer.

Used by BOTH workstreams:
  WS1 — proves query_data / retrieve produce the graded facts.
  WS2 — proves the loop + synthesis carry those facts into the final `answer`.

Usage (from agentic-team/, after `source team.env`):
  python -m scripts.check_questions                 # all 15
  python -m scripts.check_questions MHQ001 MHQ055   # a subset
  PUBLIC_QUESTIONS=/path/to/public_questions.jsonl python -m scripts.check_questions

Exit code 0 = every question hit all its numeric/date facts.
"""
from __future__ import annotations

import json
import os
import re
import sys

DEFAULT_QS = os.path.expanduser(
    "~/Cognitivo_Training/Mock_Hackathon_Participant_Package/public_questions.jsonl"
)

# tokens that carry a graded fact: numbers, signed decimals, percents, thousands, years
_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _nums(text: str) -> list[str]:
    """Salient numeric tokens, comma-stripped, deduped in order."""
    out, seen = [], set()
    for m in _NUM_RE.findall(text or ""):
        tok = m.replace(",", "").lstrip("+")
        # keep meaningful tokens (skip lone punctuation artifacts)
        if tok and tok not in seen and re.search(r"\d", tok):
            seen.add(tok)
            out.append(tok)
    return out


def _answer_has(answer: str, tok: str) -> bool:
    a = answer.replace(",", "")
    if tok in a:
        return True
    # tolerate sign / trailing-zero drift (e.g. -2.25 vs 2.25, 2.50 vs 2.5)
    bare = tok.lstrip("-")
    if bare and bare in a:
        return True
    try:
        f = float(tok)
        for cand in _nums(answer):
            if abs(float(cand) - f) < 1e-6:
                return True
    except ValueError:
        pass
    return False


def check_fact(answer: str, fact: str) -> tuple[bool, list[str]]:
    """Return (ok, missing_tokens). ok = all numeric tokens in the fact appear in answer.
    Facts with no numbers fall back to a loose keyword check."""
    toks = _nums(fact)
    if not toks:
        # no numbers — check a couple of distinctive words are present
        words = [w for w in re.findall(r"[A-Za-z%]{4,}", fact.lower())][:3]
        missing = [w for w in words if w not in answer.lower()]
        return (not missing, missing)
    missing = [t for t in toks if not _answer_has(answer, t)]
    return (not missing, missing)


def main(argv: list[str]) -> int:
    qs_path = os.environ.get("PUBLIC_QUESTIONS", DEFAULT_QS)
    if not os.path.exists(qs_path):
        print(f"!! questions file not found: {qs_path}", file=sys.stderr)
        return 2
    wanted = set(argv) or None

    from agent.graph import run  # imported here so env is already sourced

    total_q = ok_q = 0
    for line in open(qs_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        q = json.loads(line)
        qid = q.get("id", "?")
        if wanted and qid not in wanted:
            continue
        total_q += 1
        try:
            out = run(q["prompt"])
            answer = out.get("answer", "") or ""
            steps = out.get("steps", 0)
            trace = out.get("tool_trace", [])
        except Exception as e:  # noqa: BLE001
            print(f"[{qid}] EXCEPTION: {e}")
            continue

        facts = [c["expected_fact"] for c in q.get("grading", {}).get("components", [])]
        results = [(f, *check_fact(answer, f)) for f in facts]
        all_ok = all(r[1] for r in results)
        ok_q += all_ok
        tools = ",".join(t.get("name", t.get("tool", "?")) for t in trace)
        print(f"\n{'✅' if all_ok else '❌'} [{qid}] steps={steps} tools=[{tools}]")
        print(f"    answer: {answer[:240]}")
        for fact, fok, missing in results:
            mark = " ok " if fok else "MISS"
            extra = "" if fok else f"  <-- missing {missing}"
            print(f"     [{mark}] {fact}{extra}")

    print(f"\n=== {ok_q}/{total_q} questions hit every fact ===")
    return 0 if ok_q == total_q else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
