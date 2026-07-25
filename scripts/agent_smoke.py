"""Smoke test: single-tool-call questions, as a regression check on `ask`.

Run from the repo root (after copying .env.example to .env and filling in
COHERE_API_KEY):

    python scripts/agent_smoke.py

Each question here is answerable with exactly one tool call, so it should
resolve in a single loop iteration -- see scripts/loop_smoke.py for
questions that require the model to chain multiple tool calls across
rounds. Prints which tool the model chose, the arguments, the raw tool
result, and the model's final answer so tool selection can be eyeballed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import ask

QUESTIONS = [
    "What team does Erling Haaland play for and what's his price?",
    "What's the average score for gameweek 1?",
]


def main() -> None:
    for question in QUESTIONS:
        print(f"=== {question} ===")
        result = ask(question)
        if result["tool_calls"]:
            for call in result["tool_calls"]:
                print(f"  tool: {call['name']}({call['arguments']})")
                print(f"  result: {call['result']}")
        else:
            print("  (no tool called)")
        print(f"  iterations: {result['iterations']}")
        print(f"  answer: {result['answer']}\n")


if __name__ == "__main__":
    main()
