"""Tests for app/tools.py, run against the real live FPL API.

No auth needed and no cost, unlike the Cohere-backed agent -- consistent
with this project's Stage 1-3 convention of verifying the data layer
against live data rather than mocking it.
"""

from app import tools


def test_find_player_returns_team_id_in_valid_range():
    result = tools.find_player("Haaland")

    assert "team_id" in result
    assert 1 <= result["team_id"] <= 20


def test_find_player_team_id_matches_get_team_fixtures():
    result = tools.find_player("Haaland")

    fixtures = tools.get_team_fixtures(result["team_id"])

    assert "error" not in fixtures
    assert fixtures["team"] == result["team"]
