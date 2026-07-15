#!/usr/bin/env python3
"""
Evaluate base model vs finetuned model on the held-out test set.
Generates a comparison report for hackathon judges.

Usage:
    python 05_evaluate.py \
        --test_file /home/cognitivo/deploy/hackathon-finagent/data/test.jsonl \
        --base_url  http://localhost:8000/v1 \
        --ft_url    http://localhost:8001/v1 \
        --base_model qwen3.5 \
        --ft_model   nemotron-finance
"""

import json
import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

FINANCIAL_KEYWORDS = {
    "sentiment": ["bullish", "bearish", "positive", "negative", "neutral",
                  "upside", "downside", "rally", "correction", "volatile"],
    "macro":     ["rba", "cash rate", "inflation", "gdp", "asx", "aud",
                  "basis points", "yield", "monetary policy", "recession"],
    "technical": ["support", "resistance", "moving average", "volume",
                  "breakout", "trend", "momentum", "consolidation"],
    "risk":      ["risk", "exposure", "hedge", "diversif", "drawdown",
                  "volatility", "correlation", "beta"],
}


def call_model(client, model, prompt, max_tokens=128):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"ERROR: {e}"


def score_response(response, expected_output):
    response_lower = response.lower()
    expected_lower = expected_output.lower()

    scores = {}

    # Keyword coverage per category
    for category, keywords in FINANCIAL_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in response_lower)
        scores[f"{category}_keywords"] = round(hits / len(keywords), 3)

    # Expected keyword overlap
    expected_words = set(re.findall(r'\b\w{4,}\b', expected_lower))
    response_words = set(re.findall(r'\b\w{4,}\b', response_lower))
    if expected_words:
        overlap = len(expected_words & response_words) / len(expected_words)
        scores["expected_overlap"] = round(overlap, 3)
    else:
        scores["expected_overlap"] = 0

    # Length score (penalise very short responses)
    scores["length_score"] = min(1.0, len(response.split()) / 80)

    # Composite
    scores["composite"] = round(
        0.3 * scores["expected_overlap"] +
        0.2 * scores["sentiment_keywords"] +
        0.2 * scores["macro_keywords"] +
        0.15 * scores["technical_keywords"] +
        0.15 * scores["length_score"],
        3
    )
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file",   required=True)
    parser.add_argument("--base_url",    default="http://localhost:8000/v1")
    parser.add_argument("--ft_url",      default="http://localhost:8001/v1")
    parser.add_argument("--base_model",  default="qwen3.5")
    parser.add_argument("--ft_model",    default="nemotron-finance")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--out",         default="./eval_report.json")
    args = parser.parse_args()

    base_client = OpenAI(base_url=args.base_url, api_key="dummy")
    ft_client   = OpenAI(base_url=args.ft_url,   api_key="dummy")

    with open(args.test_file) as f:
        samples = [json.loads(l) for l in f][:args.max_samples]

    print(f"Evaluating {len(samples)} samples...")
    print(f"  Base model: {args.base_model} @ {args.base_url}")
    print(f"  Finetuned:  {args.ft_model}  @ {args.ft_url}")
    print()

    results = []
    base_totals = {}
    ft_totals   = {}

    for i, sample in enumerate(samples):
        prompt = sample["input"]
        expected = sample["output"]

        with ThreadPoolExecutor(max_workers=2) as ex:
            f_base = ex.submit(call_model, base_client, args.base_model, prompt)
            f_ft   = ex.submit(call_model, ft_client,   args.ft_model,   prompt)
            base_response = f_base.result()
            ft_response   = f_ft.result()

        base_scores = score_response(base_response, expected)
        ft_scores   = score_response(ft_response,   expected)

        for k in base_scores:
            base_totals[k] = base_totals.get(k, 0) + base_scores[k]
            ft_totals[k]   = ft_totals.get(k, 0)   + ft_scores[k]

        results.append({
            "prompt":        prompt[:200],
            "expected":      expected[:200],
            "base_response": base_response[:300],
            "ft_response":   ft_response[:300],
            "base_scores":   base_scores,
            "ft_scores":     ft_scores,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] base={base_scores['composite']:.3f} ft={ft_scores['composite']:.3f}")

    # Averages
    n = len(samples)
    base_avg = {k: round(v/n, 3) for k, v in base_totals.items()}
    ft_avg   = {k: round(v/n, 3) for k, v in ft_totals.items()}
    delta    = {k: round(ft_avg[k] - base_avg[k], 3) for k in base_avg}

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"{'Metric':<25} {'Base':>8} {'Finetuned':>10} {'Delta':>8}")
    print("-"*60)
    for k in sorted(base_avg.keys()):
        arrow = "✓" if delta[k] > 0 else "✗"
        print(f"{k:<25} {base_avg[k]:>8.3f} {ft_avg[k]:>10.3f} {delta[k]:>+8.3f} {arrow}")
    print("="*60)

    improvement = round((ft_avg["composite"] - base_avg["composite"]) / max(base_avg["composite"], 0.001) * 100, 1)
    print(f"\nOverall improvement: {improvement:+.1f}% on composite score")

    report = {
        "base_model": args.base_model,
        "ft_model":   args.ft_model,
        "n_samples":  n,
        "base_avg":   base_avg,
        "ft_avg":     ft_avg,
        "delta":      delta,
        "improvement_pct": improvement,
        "samples":    results,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to {args.out}")


if __name__ == "__main__":
    main()
