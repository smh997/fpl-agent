"""Standalone MCP server exposing the FPL tools over Streamable HTTP.

Stage 3 step 1: wraps the existing app/tools.py functions as MCP tools so
any MCP host (Inspector, Claude, a future agent rewire) can call them --
without touching the agent, /chat, or tools.py's own logic. Each tool below
is a thin wrapper that imports and delegates to app/tools.py; the fuzzy
matching, thresholds, and data shapes all still live there, unchanged.

Tool descriptions and per-parameter Field descriptions intentionally reuse
the wording tuned in app/schemas.py (the Gemini tool schemas) -- an MCP
host chooses tools and fills arguments the same way an LLM tool-calling API
does, so the same grounding/parameter guidance applies here.

Run standalone (defaults to 127.0.0.1:8001 so it doesn't collide with the
FastAPI backend on 8000):

    python mcp_server/server.py

Override the bind address with MCP_HOST / MCP_PORT env vars. Inspect it with
MCP Inspector (requires Node.js):

    npx @modelcontextprotocol/inspector

then connect to http://127.0.0.1:8001/mcp with transport "Streamable HTTP".

Not wired into the agent yet -- this is a standalone service for manual
testing before anything gets rewired to use it.
"""

import os
import sys
from pathlib import Path
from typing import Annotated

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server import MCPServer
from pydantic import Field

from app import tools

MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("MCP_PORT", "8001"))

mcp_server = MCPServer(
    name="fpl-agent-tools",
    instructions=(
        "Fantasy Premier League data tools: look up a player, get their "
        "season stats, get a team's upcoming fixtures, or summarize a "
        "gameweek. All data is grounded in the live FPL API -- never guess "
        "player/team/fixture facts from your own knowledge."
    ),
)


@mcp_server.tool()
def find_player(
    name: Annotated[
        str, Field(description="The player's full or partial name, e.g. 'Salah' or 'Mo Salah'.")
    ],
) -> dict:
    """Look up a Fantasy Premier League player by name. This is the
    required first step for ANY question about a specific player --
    including general questions like 'who is X' or 'tell me about X' --
    since it's the only way to ground an answer in real, current player
    data (id, team, position, price). Always call this before answering a
    question about a named player, even if you think you already know who
    they are. Matching is fuzzy and case-insensitive. The result includes
    both the team's name and its numeric team_id -- use team_id, not the
    name, when calling get_team_fixtures. Returns {"error": "no player
    found"} if nothing matches well enough; treat that as final and do not
    retry with guessed name variations.
    """
    return tools.find_player(name)


@mcp_server.tool()
def get_player_stats(
    player_id: Annotated[
        int, Field(description="The player's numeric FPL element id (e.g. from find_player).")
    ],
) -> dict:
    """Get season-to-date performance stats (form, total points, goals,
    assists, minutes, points per game) for a player, given their numeric
    player id. Requires the id -- if you only have a player's name, call
    find_player first to get their id.
    """
    return tools.get_player_stats(player_id)


@mcp_server.tool()
def get_team_fixtures(
    team_id: Annotated[int, Field(description="The team's numeric FPL team id, not its name.")],
    next_n: Annotated[
        int, Field(description="How many upcoming fixtures to return. Defaults to 5 if not specified.")
    ] = 5,
) -> dict:
    """Get a team's next upcoming (not yet played) fixtures, including
    opponent, home/away venue, and difficulty rating. Requires the team's
    numeric FPL team id, not its name.
    """
    return tools.get_team_fixtures(team_id, next_n)


@mcp_server.tool()
def get_gameweek_summary(
    gameweek: Annotated[
        int | None,
        Field(description="The gameweek number to summarize. Omit to use the current gameweek."),
    ] = None,
) -> dict:
    """Get summary stats for a Fantasy Premier League gameweek: average
    score, highest score, the most-captained player id, and the top-scoring
    player id. If no gameweek number is given, returns the current (or next
    upcoming) gameweek.
    """
    return tools.get_gameweek_summary(gameweek)


if __name__ == "__main__":
    mcp_server.run(transport="streamable-http", host=MCP_HOST, port=MCP_PORT)
