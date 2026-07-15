"""Stage 4 — minimal UI backend. FastAPI SSE endpoint streaming the contracts §4 frames:

    {"type":"token","data":"<text>"} ...  {"type":"sources","data":[Doc,...]}  {"type":"done"}

Reference simplification: the agent runs to completion, then the answer streams in word
chunks (true per-token streaming from the LLM is the documented stretch). The frame contract
the judges verify (DoD #3) is exact either way.

Run:  uvicorn ui.app:app --port 8080     (from p2_agent/; mock or real brain per env vars)
Open: http://localhost:8080
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from agent import graph

app = FastAPI(title="Sample Activity 5 — agent chat")
STATIC = Path(__file__).parent / "static"


class ChatIn(BaseModel):
    message: str


def _frames(question: str) -> Iterator[str]:
    final = graph.run(question)
    words = (final.get("answer") or "(no answer)").split(" ")
    for i in range(0, len(words), 4):
        chunk = " ".join(words[i:i + 4]) + " "
        yield f'data: {json.dumps({"type": "token", "data": chunk})}\n\n'
    yield f'data: {json.dumps({"type": "sources", "data": final.get("sources", [])})}\n\n'
    yield f'data: {json.dumps({"type": "done"})}\n\n'


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/chat")
def chat(body: ChatIn):
    return StreamingResponse(_frames(body.message), media_type="text/event-stream")
