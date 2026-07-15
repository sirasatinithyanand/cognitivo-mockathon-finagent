"""`pdf_reader` — extract text chunks from a report PDF (skill from final_day3_plan.md §3).

Local-filesystem paths only (on-prem: PDFs live in the shared data dir). pdfplumber is
imported lazily so the rest of the toolset works without it installed.
"""
from __future__ import annotations

from pathlib import Path

from .registry import register

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "path to a local PDF"},
        "max_pages": {"type": "integer", "minimum": 1, "default": 10},
        "chunk_chars": {"type": "integer", "minimum": 200, "default": 1500},
    },
    "required": ["path"],
}
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "chunks": {"type": "array", "items": {"type": "string"}},
        "pages_read": {"type": "integer"},
    },
    "required": ["chunks"],
}


@register("pdf_reader",
          "Extract text from a local PDF report and return it as chunks for analysis.",
          INPUT_SCHEMA, OUTPUT_SCHEMA)
def pdf_reader(path: str, max_pages: int = 10, chunk_chars: int = 1500) -> dict:
    import pdfplumber  # lazy: keeps offline demo dependency-light

    p = Path(path)
    if not p.exists():
        return {"chunks": [], "pages_read": 0, "error": f"no such file: {path}"}
    text_parts: list[str] = []
    with pdfplumber.open(p) as pdf:
        pages = pdf.pages[:max_pages]
        for page in pages:
            text_parts.append(page.extract_text() or "")
    full = "\n".join(text_parts)
    chunks = [full[i:i + chunk_chars] for i in range(0, len(full), chunk_chars)]
    return {"chunks": chunks, "pages_read": len(pages)}
