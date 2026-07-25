"""Cohere tool-calling schemas for the functions in app/tools.py.

These descriptions ARE prompts -- the model chooses a tool and fills in
arguments based solely on `description` text here, not on the Python
docstrings. Keep each tool's purpose and parameter meaning unambiguous so
the model doesn't confuse, e.g., a name-based lookup with an id-based one.
"""

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "find_player",
            "description": (
                "Look up a Fantasy Premier League player by name. This is the "
                "required first step for ANY question about a specific "
                "player -- including general questions like 'who is X' or "
                "'tell me about X' -- since it's the only way to ground an "
                "answer in real, current player data (id, team, position, "
                "price). Always call this before answering a question about "
                "a named player, even if you think you already know who "
                "they are. Matching is fuzzy and case-insensitive. The "
                "result includes both the team's name and its numeric "
                "team_id -- use team_id, not the name, when calling "
                "get_team_fixtures."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The player's full or partial name.",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_player_stats",
            "description": (
                "Get season-to-date performance stats (form, total points, "
                "goals, assists, minutes, points per game) for a player, "
                "given their numeric player id. Requires the id -- if you "
                "only have a player's name, call find_player first to get "
                "their id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "player_id": {
                        "type": "integer",
                        "description": "The player's numeric FPL element id.",
                    },
                },
                "required": ["player_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_team_fixtures",
            "description": (
                "Get a team's next upcoming (not yet played) fixtures, "
                "including opponent, home/away venue, and difficulty rating. "
                "Requires the team's numeric FPL team id, not its name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "team_id": {
                        "type": "integer",
                        "description": "The team's numeric FPL team id.",
                    },
                    "next_n": {
                        "type": "integer",
                        "description": (
                            "How many upcoming fixtures to return. "
                            "Defaults to 5 if not specified."
                        ),
                    },
                },
                "required": ["team_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_gameweek_summary",
            "description": (
                "Get summary stats for a Fantasy Premier League gameweek: "
                "average score, highest score, the most-captained player id, "
                "and the top-scoring player id. If no gameweek number is "
                "given, returns the current (or next upcoming) gameweek."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "gameweek": {
                        "type": "integer",
                        "description": (
                            "The gameweek number to summarize. Omit to use "
                            "the current gameweek."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
]
