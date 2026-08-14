"""Shared configuration for the agent's LLM provider and tool execution."""

import os

MODEL = "gemini-3.6-flash"

# Standalone MCP server (mcp_server/server.py) exposing the FPL tools over
# Streamable HTTP. Only used when ask(..., use_mcp=True) opts in -- the
# default direct-call path never reads this.
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8001/mcp")
