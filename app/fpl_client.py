"""Thin HTTP client over the public Fantasy Premier League API.

Base URL: https://fantasy.premierleague.com/api/. This is an unofficial,
undocumented API with no auth on the endpoints used here. This module only
fetches and parses JSON responses -- it does not filter or shape data for
callers (see app/tools.py for that).
"""

import httpx

BASE_URL = "https://fantasy.premierleague.com/api/"
TIMEOUT_SECONDS = 10.0

_client = httpx.Client(base_url=BASE_URL, timeout=TIMEOUT_SECONDS)

_bootstrap_cache: dict | None = None


def get_bootstrap() -> dict:
    """Fetch bootstrap-static/, the FPL API's main reference dataset.

    Contains all players (elements), teams, positions (element_types), and
    gameweeks (events). This payload is large (~1-2 MB) and several tools
    depend on it, so it is fetched once and cached in memory for the
    lifetime of the process.

    Returns:
        The parsed JSON response as a dict.
    """
    global _bootstrap_cache
    if _bootstrap_cache is None:
        response = _client.get("bootstrap-static/")
        response.raise_for_status()
        _bootstrap_cache = response.json()
    return _bootstrap_cache


def get_element_summary(player_id: int) -> dict:
    """Fetch element-summary/{player_id}/, a single player's detailed history.

    Args:
        player_id: The FPL element id of the player.

    Returns:
        The parsed JSON response as a dict, including past fixtures,
        upcoming fixtures, and season-by-season history.
    """
    response = _client.get(f"element-summary/{player_id}/")
    response.raise_for_status()
    return response.json()


def get_fixtures(gameweek: int | None = None) -> list[dict]:
    """Fetch fixtures/, the full season fixture list.

    Args:
        gameweek: If given, restrict results to this gameweek (event) only
            via the `?event=` query parameter. If None, returns all
            fixtures for the season.

    Returns:
        The parsed JSON response: a list of fixture dicts.
    """
    params = {"event": gameweek} if gameweek is not None else None
    response = _client.get("fixtures/", params=params)
    response.raise_for_status()
    return response.json()
