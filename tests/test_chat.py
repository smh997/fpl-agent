"""Tests for the POST /chat endpoint's HTTP contract.

app.agent.ask is monkeypatched throughout -- these tests verify request
validation, status codes, and response shaping, not live Cohere behavior
(that's covered by scripts/agent_smoke.py and scripts/loop_smoke.py against
the real API).
"""

from fastapi.testclient import TestClient

from app import agent
from app.main import app

client = TestClient(app)


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
    }
    monkeypatch.setattr(agent, "ask", lambda question, max_iterations: canned_result)

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


def test_chat_returns_503_when_api_key_missing(monkeypatch):
    def fake_ask(question, max_iterations):
        raise RuntimeError(
            "COHERE_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    monkeypatch.setattr(agent, "ask", fake_ask)

    response = client.post("/chat", json={"question": "any question"})

    assert response.status_code == 503
    assert "COHERE_API_KEY" in response.json()["detail"]
