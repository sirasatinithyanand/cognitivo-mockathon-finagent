"""HTTP wrapper — teams run this so the eval harness can reach their agent.

Usage:
    python server.py            # port 5000 default
    PORT=5001 python server.py  # custom port

POST /query  {"question": "..."}
-> {"answer": "...", "steps": N, "tool_trace": [...]}
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent.graph import run

app = FastAPI(title="Cognitivo Financial Agent")


class Query(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query(q: Query):
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="question is empty")
    state = run(q.question)
    return {
        "answer": state.get("answer", ""),
        "steps": state.get("steps", 0),
        "tool_trace": state.get("tool_trace", []),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
