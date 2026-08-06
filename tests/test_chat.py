"""Tests for the POST /chat endpoint's HTTP contract.

app.agent.ask_with_fallback is monkeypatched for the happy path; the
resilience tests below monkeypatch the lower-level app.agent.ask instead so
the real ask_with_fallback fallback/error-shaping logic runs. Neither
exercises live Gemini behavior (that's covered by scripts/agent_smoke.py,
scripts/loop_smoke.py, and scripts/capture_fixtures.py against the real
API).
"""

import httpx
from fastapi.testclient import TestClient
from google.genai._gaos.lib.compat_errors import RateLimitError

from app import agent
from app.main import app

client = TestClient(app)


def _rate_limit_error() -> RateLimitError:
    request = httpx.Request("POST", "https://example.invalid/interactions")
    response = httpx.Response(429, request=request, json={"error": {"message": "rate limited"}})
    return RateLimitError("rate limited", response=response, body={"error": {"message": "rate limited"}})


def test_chat_happy_path(monkeypatch):
    canned_result = {
        "question": "What team does Erling Haaland play for?",
        "tool_calls": [
            {
                "name": "find_player",
                "arguments": {"name": "Erling Haaland"},
                "result": {"id": 411, "name": "Haaland", "team": "Man City"},
                "iteration": 1,
            }
        ],
        "answer": "Erling Haaland plays for Man City.",
        "iterations": 2,
        "demo_mode": False,
    }
    monkeypatch.setattr(
        agent, "ask_with_fallback", lambda question, max_iterations: canned_result
    )

    response = client.post("/chat", json={"question": "What team does Erling Haaland play for?"})

    assert response.status_code == 200
    assert response.json() == canned_result


def test_chat_rejects_empty_question():
    response = client.post("/chat", json={"question": ""})

    assert response.status_code == 422


def test_chat_rejects_max_iterations_out_of_bounds():
    too_low = client.post("/chat", json={"question": "x", "max_iterations": 0})
    too_high = client.post("/chat", json={"question": "x", "max_iterations": 11})

    assert too_low.status_code == 422
    assert too_high.status_code == 422


def test_chat_falls_back_to_fixture_on_rate_limit(monkeypatch):
    """When the live call exhausts retries on a 429, a captured demo
    question should be answered from fixtures/demo_fixtures.json instead of
    failing, with demo_mode reporting the switch."""
    fixtures = agent._load_fixtures()
    assert fixtures, "fixtures/demo_fixtures.json is empty -- run scripts/capture_fixtures.py"
    question, cached = next(iter(fixtures.items()))

    monkeypatch.setattr(agent, "ask", lambda q, max_iterations: (_ for _ in ()).throw(_rate_limit_error()))

    response = client.post("/chat", json={"question": cached["question"]})

    assert response.status_code == 200
    body = response.json()
    assert body["demo_mode"] is True
    assert body["answer"] == cached["answer"]
    assert body["tool_calls"] == cached["tool_calls"]


def test_chat_returns_clean_error_when_uncaptured_and_live_call_fails(monkeypatch):
    """When the live call fails and the question has no matching fixture,
    /chat should return a clean error dict rather than crash or 500."""
    monkeypatch.setattr(agent, "ask", lambda q, max_iterations: (_ for _ in ()).throw(_rate_limit_error()))

    response = client.post(
        "/chat", json={"question": "This question is not in any fixture, guaranteed"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["tool_calls"] == []
    assert body["answer"] is None
    assert body["error"]
    assert body["demo_mode"] is False


def test_chat_falls_back_gracefully_when_api_key_missing(monkeypatch):
    """A missing API key is one of ask_with_fallback's specific catch cases
    (alongside rate-limit-after-retries and connection errors), so it now
    degrades gracefully instead of the old hard 503 -- this documents that
    behavior change from the pre-fallback implementation."""

    def fake_ask(question, max_iterations):
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    monkeypatch.setattr(agent, "ask", fake_ask)

    response = client.post(
        "/chat", json={"question": "This question is not in any fixture, guaranteed"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["tool_calls"] == []
    assert body["answer"] is None
    assert "GEMINI_API_KEY" in body["error"]
    assert body["demo_mode"] is False
