"""One-time capture: run demo questions through the live agent and save
their full results to fixtures/demo_fixtures.json.

app.agent.ask_with_fallback serves these when a live Gemini call fails (rate
limit exhausted, missing/invalid key, connection error) so the deployed app
can still answer these specific questions instead of erroring out.

Run from the repo root (after copying .env.example to .env and filling in
GEMINI_API_KEY):

    python scripts/capture_fixtures.py

Re-run whenever the demo questions below change, or whenever FPL data has
moved on enough that the cached answers would look stale during a demo.
fixtures/demo_fixtures.json is committed to the repo -- it holds only
public FPL data and the agent's own tool-call traces, nothing secret.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import ask

QUESTIONS = [
    "What team does Erling Haaland play for and what's his price?",
    "Compare Haaland and Saka's points",
    "Is Haaland worth captaining?",
    "Show Arsenal's next five fixtures",
    "What's the average score for gameweek 1?",
]

FIXTURES_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "demo_fixtures.json"


def _save(fixtures: dict[str, dict]) -> None:
    FIXTURES_PATH.parent.mkdir(exist_ok=True)
    FIXTURES_PATH.write_text(
        json.dumps(fixtures, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    fixtures: dict[str, dict] = {}
    if FIXTURES_PATH.exists():
        fixtures = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))

    for question in QUESTIONS:
        key = question.strip().lower()
        if key in fixtures:
            print(f"=== {question} === (already captured, skipping)")
            continue
        print(f"=== {question} ===")
        result = ask(question)
        fixtures[key] = result
        _save(fixtures)  # save after each question so a later failure doesn't lose progress
        print(f"  iterations: {result['iterations']}")
        print(f"  answer: {result['answer']}\n")

    print(f"Saved {len(fixtures)} fixtures to {FIXTURES_PATH}")


if __name__ == "__main__":
    main()
