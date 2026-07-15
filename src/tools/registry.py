"""Tool registry — one typed definition per skill (contracts §2), exported to two surfaces:
LangGraph binding (OpenAI tools format) and the Stage-3 MCP server.

Tool functions use explicit typed keyword parameters so FastMCP can introspect their
signatures. Importing `tools` (the package) registers everything via tools/__init__.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    output_schema: dict
    fn: Callable[..., dict]


_REGISTRY: dict[str, Tool] = {}


def register(name: str, description: str, input_schema: dict, output_schema: dict):
    def deco(fn: Callable[..., dict]):
        _REGISTRY[name] = Tool(name, description, input_schema, output_schema, fn)
        return fn
    return deco


def get(name: str) -> Tool:
    return _REGISTRY[name]


def all_tools() -> list[Tool]:
    return list(_REGISTRY.values())


def call(name: str, args: dict[str, Any]) -> dict:
    return _REGISTRY[name].fn(**args)


def to_openai_tools() -> list[dict]:
    """OpenAI/LiteLLM `tools=` payload — what the LangGraph reason node binds."""
    return [
        {"type": "function",
         "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
        for t in _REGISTRY.values()
    ]
