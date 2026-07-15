"""Memory interface — contracts §3, the single retrieval signature P2 depends on:

    query(text, k=8, filters=None) -> list[Doc]
    Doc = {id, text, score, source, metadata}

Backends:
  fixture (QDRANT_URL unset) : keyword-overlap ranking over mocks/fixtures/docs.json —
                               deterministic, offline, good enough to exercise the loop.
  qdrant  (QDRANT_URL set)   : embed via LiteLLM `embed` alias, search the collection
                               P3's ingestion glue populates (p3_tasks.md week 2).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from agent import config

FIXTURES = Path(__file__).resolve().parents[1] / "mocks" / "fixtures" / "docs.json"


@dataclass
class Doc:
    id: str
    text: str
    score: float
    source: str
    metadata: dict = field(default_factory=dict)


def query(text: str, k: int = 8, filters: dict | None = None) -> list[Doc]:
    if config.QDRANT_URL:
        return _query_qdrant(text, k, filters)
    return _query_fixture(text, k, filters)


def _query_fixture(text: str, k: int, filters: dict | None) -> list[Doc]:
    docs = json.loads(FIXTURES.read_text(encoding="utf-8"))
    terms = set(re.findall(r"[a-z0-9]+", text.lower()))
    scored = []
    for d in docs:
        if filters and any(d.get("metadata", {}).get(fk) != fv for fk, fv in filters.items()):
            continue
        words = set(re.findall(r"[a-z0-9]+", d["text"].lower()))
        overlap = len(terms & words) / (len(terms) or 1)
        scored.append(Doc(d["id"], d["text"], round(overlap, 4), d["source"], d.get("metadata", {})))
    scored.sort(key=lambda d: d.score, reverse=True)
    return scored[:k]


def _query_qdrant(text: str, k: int, filters: dict | None) -> list[Doc]:
    from openai import OpenAI
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    emb_client = OpenAI(base_url=config.LITELLM_BASE_URL, api_key=config.LITELLM_KEY)
    vector = emb_client.embeddings.create(model=config.EMBED_MODEL, input=text).data[0].embedding

    qfilter = None
    if filters:
        qfilter = Filter(must=[
            FieldCondition(key=f"metadata.{fk}", match=MatchValue(value=fv))
            for fk, fv in filters.items()
        ])
    hits = QdrantClient(url=config.QDRANT_URL).search(
        collection_name=config.QDRANT_COLLECTION, query_vector=vector,
        query_filter=qfilter, limit=k, with_payload=True)
    return [
        Doc(str(h.id), h.payload.get("text", ""), float(h.score),
            h.payload.get("source", "qdrant"), h.payload.get("metadata", {}))
        for h in hits
    ]
