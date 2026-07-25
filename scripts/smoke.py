"""Smoke test: exercise all four tool functions against the live FPL API.

Run from the repo root:

    python scripts/smoke.py

Prints each tool's output so results can be eyeballed against the live FPL
site. This is Stage 1's verification method -- no agent/LLM involved yet.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import fpl_client, tools


def main() -> None:
    # Salah is not in the live dataset as of this snapshot (transferred out /
    # not registered this season) -- kept here to show find_player correctly
    # returns "no player found" instead of a bogus fuzzy match.
    print("=== find_player('Salah') ===")
    print(tools.find_player("Salah"))

    print("\n=== find_player('Haaland') ===")
    player = tools.find_player("Haaland")
    print(player)

    if "error" in player:
        print("Stopping: find_player failed, cannot chain into later steps.")
        return

    print("\n=== get_player_stats(player['id']) ===")
    stats = tools.get_player_stats(player["id"])
    print(stats)

    print("\n=== get_team_fixtures(team_id, next_n=5) ===")
    bootstrap_teams = {
        team["name"]: team["id"] for team in fpl_client.get_bootstrap()["teams"]
    }
    team_id = bootstrap_teams[player["team"]]
    fixtures = tools.get_team_fixtures(team_id, next_n=5)
    print(fixtures)

    print("\n=== get_gameweek_summary() ===")
    gw_summary = tools.get_gameweek_summary()
    print(gw_summary)


if __name__ == "__main__":
    main()
