"""`recommender` — comparable-stocks suggestion (skill from final_day3_plan.md §3).

Offline: static sector-similarity table (deterministic). Real stack: nearest-neighbour
search over ticker profile embeddings in Qdrant (same query_client backend) — swap kept
behind the same tool signature.
"""
from __future__ import annotations

from .registry import register

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "ticker": {"type": "string"},
        "k": {"type": "integer", "minimum": 1, "default": 3},
    },
    "required": ["ticker"],
}
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "similar": {"type": "array",
                    "items": {"type": "object",
                              "properties": {"ticker": {"type": "string"},
                                             "score": {"type": "number"}}}},
    },
    "required": ["similar"],
}

# offline stand-in: sector neighbours with fixed similarity scores
_SIMILAR = {
    "BHP": [("RIO", 0.94), ("FMG", 0.88), ("S32", 0.81)],
    "CBA": [("WBC", 0.93), ("NAB", 0.92), ("ANZ", 0.90)],
    "CSL": [("RMD", 0.77), ("COH", 0.74), ("SHL", 0.70)],
}


@register("recommender",
          "Suggest comparable ASX stocks for a ticker (ranked by similarity).",
          INPUT_SCHEMA, OUTPUT_SCHEMA)
def recommender(ticker: str, k: int = 3) -> dict:
    pairs = _SIMILAR.get(ticker.upper(), [])[:k]
    return {"similar": [{"ticker": t, "score": s} for t, s in pairs]}
