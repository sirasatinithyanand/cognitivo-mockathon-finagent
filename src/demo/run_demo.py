"""End-to-end demo query — the DoD #2 proof, runnable offline.

Runs the LangGraph agent on the demo question and asserts the answer is *agentic*:
  >= 1 skill call (domain_predict) AND >= 1 retrieval (retrieve), with a non-empty answer.
Prints the tool trace (judges' workflow-reasoning evidence), the sources, and the answer.
Exit code 0 = pass.

Usage (from p2_agent/):
  python -m demo.run_demo --mock       # spawns mocks/mock_llm.py on :9000, runs, tears down
  python -m demo.run_demo              # against whatever LITELLM_BASE_URL points at
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEMO_QUESTION = ("Given current interest rates, should I hold BHP? "
                 "Use the model's 90-day forecast and cite recent filings.")


def start_mock() -> subprocess.Popen:
    proc = subprocess.Popen([sys.executable, "-m", "mocks.mock_llm"], cwd=ROOT)
    for _ in range(50):
        try:
            if httpx.get("http://localhost:9000/health", timeout=1).status_code == 200:
                return proc
        except httpx.HTTPError:
            time.sleep(0.2)
    proc.terminate()
    raise RuntimeError("mock LLM did not come up on :9000")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="spawn the mock LLM for the run")
    ap.add_argument("--question", default=DEMO_QUESTION)
    args = ap.parse_args()

    mock_proc = start_mock() if args.mock else None
    try:
        from agent import graph  # import after mock is up (client built per call anyway)
        final = graph.run(args.question)
    finally:
        if mock_proc:
            mock_proc.terminate()

    print("\n--- tool trace ---")
    for t in final["tool_trace"]:
        print(f"  {t['name']}({t['args']}) [{t['ms']}ms] -> {t['summary']}")
    print("\n--- sources ---")
    for s in final["sources"]:
        print(f"  [{s['id']}] {s['source']} (score={s['score']})")
    print("\n--- answer ---\n" + (final["answer"] or "(empty)"))

    tool_names = {t["name"] for t in final["tool_trace"]}
    checks = {
        ">=1 skill call": bool(tool_names - {"retrieve"}),
        ">=1 retrieval": "retrieve" in tool_names,
        "non-empty answer": bool(final["answer"].strip()),
        "sources captured": bool(final["sources"]),
    }
    print("\n--- DoD checks ---")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
