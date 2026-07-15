"""Stage 3 — MCP server exposing the SAME tool registry the agent binds locally.

One tool definition, two surfaces: tools/registry.py entries are added to a FastMCP server,
so `domain_predict`, `retrieve`, `pdf_reader`, `recommender` become MCP tools with no
duplicate schemas. The agent consumes them via langchain-mcp-adapters (integ week); any MCP
client (Claude Desktop, inspector, another team's agent) can too.

Run:
  python -m mcp_server.server                 # stdio transport (MCP default)
  python -m mcp_server.server --sse           # SSE transport on :8010
  python -m mcp_server.server --list-tools    # print registered tools and exit
"""
from __future__ import annotations

import argparse

from mcp.server.fastmcp import FastMCP

from tools import registry  # importing tools/ registers every skill

mcp = FastMCP("sample5-tools", host="127.0.0.1", port=8010)

for _tool in registry.all_tools():
    mcp.add_tool(_tool.fn, name=_tool.name, description=_tool.description)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sse", action="store_true", help="serve over SSE on :8010 (default: stdio)")
    ap.add_argument("--list-tools", action="store_true", help="print registered tools and exit")
    args = ap.parse_args()

    if args.list_tools:
        import asyncio
        tools = asyncio.run(mcp.list_tools())
        for t in tools:
            print(f"{t.name}: {t.description}")
        print(f"({len(tools)} MCP tools registered)")
        return

    mcp.run(transport="sse" if args.sse else "stdio")


if __name__ == "__main__":
    main()
