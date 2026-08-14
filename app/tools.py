"""Tool functions for the FPL agent.

Each function is a composable primitive intended to be exposed to a
tool-calling LLM in a later stage. Every function returns a small, filtered
dict -- never a raw FPL API object -- to keep tool outputs cheap in tokens
and easy for a model to reason about. Docstrings double as the basis for
future tool-schema descriptions, so they describe inputs/outputs precisely.
"""

from difflib import SequenceMatcher

from app import fpl_client

_MATCH_ACCEPT_THRESHOLD = 0.5
_TYPO_RATIO_THRESHOLD = 0.78
_MIN_SUBSTRING_QUERY_LENGTH = 3


def _match_score(query: str, candidate: str) -> float:
    """Score how well `query` identifies `candidate` (both compared lowercase).

    Exact matches score 1.0. A query that's a substring of the candidate (or
    vice versa -- e.g. "salah" in "mohamed salah") scores highly, favoring
    fuller coverage of the shorter string -- but only once the query is at
    least _MIN_SUBSTRING_QUERY_LENGTH characters, since a 1-2 character
    fragment (e.g. "mo") is a near-universal substring of *something* and
    would otherwise confidently "match" almost any name. Anything else falls
    back to a typo-tolerant SequenceMatcher ratio, discounted so that
    coincidental similarity (e.g. "salah" vs "saliba") can't outscore a real
    substring match or masquerade as a confident one.
    """
    query, candidate = query.lower(), candidate.lower()
    if query == candidate:
        return 1.0
    if len(query) >= _MIN_SUBSTRING_QUERY_LENGTH and (query in candidate or candidate in query):
        shorter, longer = sorted((len(query), len(candidate)))
        return 0.8 + 0.2 * (shorter / longer)
    ratio = SequenceMatcher(None, query, candidate).ratio()
    return ratio * 0.7 if ratio >= _TYPO_RATIO_THRESHOLD else 0.0


def find_player(name: str) -> dict:
    """Find a player by name using forgiving, case-insensitive fuzzy matching.

    Matches against each player's short "web name" (e.g. "Salah") and full
    name (e.g. "Mohamed Salah"), so partial or common names resolve to the
    right player. If several players are plausible matches, the closest one
    is returned.

    Args:
        name: A player name or partial name, e.g. "Salah" or "Mo Salah".

    Returns:
        {id, name, team, team_id, position, price} for the best match, or
        {"error": "no player found"} if nothing matches well enough.
    """
    bootstrap = fpl_client.get_bootstrap()
    teams_by_id = {team["id"]: team["name"] for team in bootstrap["teams"]}
    positions_by_id = {
        pos["id"]: pos["singular_name_short"] for pos in bootstrap["element_types"]
    }

    best_element = None
    best_score = 0.0
    for element in bootstrap["elements"]:
        full_name = f"{element['first_name']} {element['second_name']}"
        score = max(
            _match_score(name, element["web_name"]),
            _match_score(name, full_name),
        )
        if score > best_score:
            best_score = score
            best_element = element

    if best_element is None or best_score < _MATCH_ACCEPT_THRESHOLD:
        return {"error": "no player found"}

    return {
        "id": best_element["id"],
        "name": best_element["web_name"],
        "team": teams_by_id.get(best_element["team"], "unknown"),
        "team_id": best_element["team"],
        "position": positions_by_id.get(best_element["element_type"], "unknown"),
        "price": best_element["now_cost"] / 10,
    }


def get_player_stats(player_id: int) -> dict:
    """Get season-to-date performance stats for a player.

    Args:
        player_id: The FPL element id of the player (e.g. from find_player).

    Returns:
        {name, form, total_points, goals_scored, assists, minutes,
        points_per_game}, or {"error": "player not found"} if the id is
        invalid.
    """
    bootstrap = fpl_client.get_bootstrap()
    element = next(
        (e for e in bootstrap["elements"] if e["id"] == player_id), None
    )
    if element is None:
        return {"error": "player not found"}

    return {
        "name": element["web_name"],
        "form": float(element["form"]),
        "total_points": element["total_points"],
        "goals_scored": element["goals_scored"],
        "assists": element["assists"],
        "minutes": element["minutes"],
        "points_per_game": float(element["points_per_game"]),
    }


def get_team_fixtures(team_id: int, next_n: int = 5) -> dict:
    """Get a team's next upcoming (unfinished) fixtures.

    Args:
        team_id: The FPL team id.
        next_n: How many upcoming fixtures to return (default 5).

    Returns:
        {team, fixtures: [{opponent, venue, difficulty}, ...]} where venue
        is "home" or "away" and difficulty is the FPL difficulty rating
        (1-5) for that fixture from this team's perspective. Returns
        {"error": "team not found"} if the team id is invalid.
    """
    bootstrap = fpl_client.get_bootstrap()
    teams_by_id = {team["id"]: team["name"] for team in bootstrap["teams"]}
    if team_id not in teams_by_id:
        return {"error": "team not found"}

    all_fixtures = fpl_client.get_fixtures()
    upcoming = [
        fixture
        for fixture in all_fixtures
        if not fixture["finished"]
        and (fixture["team_h"] == team_id or fixture["team_a"] == team_id)
    ]
    upcoming.sort(key=lambda f: (f["event"] is None, f["event"], f["kickoff_time"]))

    fixtures = []
    for fixture in upcoming[:next_n]:
        is_home = fixture["team_h"] == team_id
        opponent_id = fixture["team_a"] if is_home else fixture["team_h"]
        difficulty = (
            fixture["team_h_difficulty"] if is_home else fixture["team_a_difficulty"]
        )
        fixtures.append(
            {
                "opponent": teams_by_id.get(opponent_id, "unknown"),
                "venue": "home" if is_home else "away",
                "difficulty": difficulty,
            }
        )

    return {"team": teams_by_id[team_id], "fixtures": fixtures}


def get_gameweek_summary(gameweek: int | None = None) -> dict:
    """Get summary stats for a gameweek.

    Args:
        gameweek: The gameweek (event) number. If None, uses the current
            gameweek (or the next upcoming one if none is currently active,
            e.g. during pre-season).

    Returns:
        {gameweek, average_score, highest_score, most_captained_id,
        top_element_id}, or {"error": "gameweek not found"} if the given
        gameweek doesn't exist and none could be inferred.
    """
    bootstrap = fpl_client.get_bootstrap()
    events = bootstrap["events"]

    if gameweek is None:
        event = next((e for e in events if e["is_current"]), None)
        if event is None:
            event = next((e for e in events if e["is_next"]), None)
    else:
        event = next((e for e in events if e["id"] == gameweek), None)

    if event is None:
        return {"error": "gameweek not found"}

    return {
        "gameweek": event["id"],
        "average_score": event["average_entry_score"],
        "highest_score": event["highest_score"],
        "most_captained_id": event["most_captained"],
        "top_element_id": event["top_element"],
    }
