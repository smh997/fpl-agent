"""Stage 3: the agent loop.

Wires the tool schemas in app/schemas.py to the real functions in
app/tools.py via Cohere's ClientV2 chat API. `ask` lets the model call
tools, see their results, and call further tools based on what it learns
(e.g. find_player to get an id, then get_player_stats with that id) --
repeating until it has enough to answer, capped at `max_iterations` so a
model that keeps requesting tools can't loop forever.
"""

import json
import os

import cohere
from dotenv import load_dotenv

from app import tools
from app.schemas import TOOL_SCHEMAS

load_dotenv()

MODEL = "command-a-03-2025"

SYSTEM_PREAMBLE = (
    "You are an FPL (Fantasy Premier League) data assistant. You do not have "
    "up-to-date knowledge of players, teams, fixtures, or statistics -- your "
    "training data is stale and FPL data changes every season. For ANY "
    "question about a specific player, team, fixture, or gameweek -- "
    "including general questions like 'who is X' -- you MUST answer using "
    "the provided tools, never from your own memory. If the tools can't "
    "answer a question, say so briefly instead of guessing or adding "
    "unrelated information. Keep answers short and stick to what the tool "
    "results actually say -- no padding, no speculation, no biographical "
    "detail the tools didn't return.\n\n"
    "When the question asks for a judgment -- 'is X worth captaining?', "
    "'should I pick A or B?', 'is X worth it?' -- commit to a clear "
    "recommendation. Don't just lay out the data and leave the decision to "
    "the user. Justify the recommendation using the retrieved tool data "
    "(form, total points, points per game, upcoming fixture difficulty) "
    "and keep the justification brief -- one or two sentences of reasoning, "
    "not a full breakdown of every stat."
)

_TOOL_REGISTRY = {
    "find_player": tools.find_player,
    "get_player_stats": tools.get_player_stats,
    "get_team_fixtures": tools.get_team_fixtures,
    "get_gameweek_summary": tools.get_gameweek_summary,
}

_client: cohere.ClientV2 | None = None


def _extract_text(content) -> str:
    """Normalize a Cohere v2 message's content (str or content-item list) to plain text."""
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return "".join(getattr(item, "text", "") for item in content)


def _get_client() -> cohere.ClientV2:
    """Lazily construct the Cohere client, failing clearly if no API key is set."""
    global _client
    if _client is None:
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "COHERE_API_KEY is not set. Copy .env.example to .env and "
                "add your key."
            )
        _client = cohere.ClientV2(api_key=api_key)
    return _client


def ask(question: str, max_iterations: int = 5) -> dict:
    """Answer a question, letting the model call tools across multiple rounds.

    Each round: the model may request one or more tool calls, which are
    executed locally and fed back as results; the model then either answers
    or requests more tools based on what it learned. This is what allows
    multi-step questions like "how many goals has Haaland scored" (needs
    find_player for the id, then get_player_stats) or comparisons between
    two players (multiple find_player/get_player_stats calls) to resolve
    without a dedicated tool for each case.

    Args:
        question: A natural-language question about FPL data.
        max_iterations: Maximum number of tool-calling rounds before giving
            up rather than looping forever.

    Returns:
        {question, tool_calls: [{name, arguments, result, iteration}, ...],
        answer, iterations}. If the model still wants more tool calls after
        max_iterations rounds, `answer` is None and an `error` key explains
        the loop was stopped.
    """
    client = _get_client()
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PREAMBLE},
        {"role": "user", "content": question},
    ]
    all_tool_calls = []

    for iteration in range(1, max_iterations + 1):
        response = client.chat(model=MODEL, messages=messages, tools=TOOL_SCHEMAS)
        tool_calls = response.message.tool_calls or []

        if not tool_calls:
            return {
                "question": question,
                "tool_calls": all_tool_calls,
                "answer": _extract_text(response.message.content),
                "iterations": iteration,
            }

        messages.append(
            {
                "role": "assistant",
                "tool_calls": tool_calls,
                "tool_plan": response.message.tool_plan,
            }
        )
        for call in tool_calls:
            arguments = json.loads(call.function.arguments)
            result = _TOOL_REGISTRY[call.function.name](**arguments)
            all_tool_calls.append(
                {
                    "name": call.function.name,
                    "arguments": arguments,
                    "result": result,
                    "iteration": iteration,
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    # Wrapped under "result" because Cohere auto-parses tool
                    # content as JSON and treats a top-level "id" key as a
                    # document id requiring a string -- which collides with
                    # tool results like find_player's integer player id.
                    "content": [
                        {"type": "text", "text": json.dumps({"result": result})}
                    ],
                }
            )

    return {
        "question": question,
        "tool_calls": all_tool_calls,
        "answer": None,
        "error": f"stopped after {max_iterations} iterations without a final answer",
        "iterations": max_iterations,
    }
