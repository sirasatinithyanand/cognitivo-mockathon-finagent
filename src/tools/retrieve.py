"""`retrieve` — the memory interface as a tool. Thin wrapper over rag.query_client.query()
(contracts §3). Doc payloads flow into AgentState.sources and the UI `sources[]` frame."""
from __future__ import annotations

from dataclasses import asdict

from rag import query_client

from .registry import register

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "k": {"type": "integer", "minimum": 1, "default": 8},
        "filters": {"type": "object",
                    "description": 'optional metadata filters, e.g. {"corpus": "law"}'},
    },
    "required": ["query"],
}
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"docs": {"type": "array", "items": {"type": "object"}}},
    "required": ["docs"],
}


@register("retrieve",
          "Semantic retrieval over the team corpora (finance filings/news/policies + law). "
          "Returns ranked documents with sources. Use filters={'corpus':'law'} for legal questions.",
          INPUT_SCHEMA, OUTPUT_SCHEMA)
def retrieve(query: str, k: int = 8, filters: dict | None = None) -> dict:
    docs = query_client.query(query, k=k, filters=filters)
    return {"docs": [asdict(d) for d in docs]}
