"""Smoke test: Stage 3's multi-tool agent loop (chaining across iterations).

Run from the repo root (after copying .env.example to .env and filling in
GEMINI_API_KEY):

    python scripts/loop_smoke.py

Unlike scripts/agent_smoke.py, these questions can't be answered with a
single tool call -- the model has to see one tool's result (e.g. a player
id from find_player) before it knows what to call next. There's
deliberately no compare_players tool: comparison questions are answered by
the model chaining find_player/get_player_stats itself.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import ask


def print_result(result: dict) -> None:
    for call in result["tool_calls"]:
        print(f"  [iter {call['iteration']}] {call['name']}({call['arguments']})")
        print(f"    -> {call['result']}")
    print(f"  iterations used: {result['iterations']}")
    if result.get("error"):
        print(f"  error: {result['error']}")
    print(f"  answer: {result['answer']}\n")


def main() -> None:
    print("=== Chained lookup: name -> id -> stats ===")
    print_result(ask("How many goals has Erling Haaland scored this season?"))

    print("=== Comparison (no compare_players tool -- model chains 2 players) ===")
    print_result(
        ask("Compare Erling Haaland and Bukayo Saka's total points this season.")
    )

    print("=== Cap safety: same chaining question, max_iterations=1 ===")
    print_result(
        ask("How many goals has Erling Haaland scored this season?", max_iterations=1)
    )


if __name__ == "__main__":
    main()
