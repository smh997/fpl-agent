"""Stage 3: the agent loop.

Wires the tool schemas in app/schemas.py to the real functions in
app/tools.py via the Gemini Interactions API. `ask` lets the model call
tools, see their results, and call further tools based on what it learns
(e.g. find_player to get an id, then get_player_stats with that id) --
repeating until it has enough to answer, capped at `max_iterations` so a
model that keeps requesting tools can't loop forever.

The loop runs stateless (store=False): rather than relying on Gemini to
retain conversation state server-side, the full step history is resent as
`input` on every call, and every step Gemini returns (plus a
function_result step per executed tool) is appended before the next call.
"""

import asyncio
import json
import os
import random
import re
import time
from pathlib import Path

import httpx2
from google import genai
from google.genai._gaos.lib.compat_errors import (
    APIConnectionError,
    AuthenticationError,
    RateLimitError,
)
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from app import tools
from app.config import MCP_SERVER_URL, MODEL
from app.schemas import TOOL_SCHEMAS

load_dotenv()

_MAX_RETRIES = 3
_MAX_BACKOFF_SECONDS = 10
_RETRY_DELAY_PATTERN = re.compile(r"retry in ([\d.]+)\s*s", re.IGNORECASE)

_FIXTURES_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "demo_fixtures.json"

SYSTEM_PREAMBLE = (
    "Your scope is strictly Fantasy Premier League: players, teams, "
    "fixtures, and gameweeks, answered using the provided tools. For any "
    "question outside that scope -- including general-knowledge questions "
    "you could answer from memory, like capitals, history, or "
    "definitions -- decline briefly and say it's outside what you can help "
    "with here. Do not answer non-FPL questions from your own knowledge, "
    "even if you're confident you know the answer.\n\n"
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
    "If find_player returns {\"error\": \"no player found\"} (or any "
    "not-found result) for a player name, stop searching for that player "
    "immediately. Do not retry find_player with alternate spellings, "
    "nicknames, shortened names, or other guessed variations of the name. "
    "Make exactly one find_player attempt per player name the user actually "
    "mentioned; if that attempt doesn't find them, tell the user you "
    "couldn't find that player and move on -- do not guess who they might "
    "be.\n\n"
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

_client: genai.Client | None = None
_fixtures_cache: dict[str, dict] | None = None


def _get_client() -> genai.Client:
    """Lazily construct the Gemini client, failing clearly if no API key is set."""
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and "
                "add your key."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def _parse_retry_delay(error: RateLimitError) -> float | None:
    """Extract a server-suggested retry delay (seconds) from a 429, if present.

    Checks the standard Retry-After header first, then falls back to
    scanning the error body's message text for Google's "Please retry in
    Xs" phrasing. Returns None if neither is present, so the caller falls
    back to exponential backoff.
    """
    response = getattr(error, "response", None)
    retry_after = response.headers.get("Retry-After") if response is not None else None
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass

    body = getattr(error, "body", None)
    message = body.get("error", {}).get("message", "") if isinstance(body, dict) else str(error)
    match = _RETRY_DELAY_PATTERN.search(message)
    return float(match.group(1)) if match else None


def _create_interaction(client: genai.Client, *, max_retries: int = _MAX_RETRIES, **kwargs):
    """Call client.interactions.create, retrying on 429s with bounded backoff.

    Up to max_retries retries (max_retries + 1 attempts total). Each delay is
    either the server-suggested retry delay (if Gemini reports one) or an
    exponential backoff with jitter, capped at _MAX_BACKOFF_SECONDS so a
    single sleep never blocks the request longer than that -- a long
    reported delay is better handled by failing over to a demo fixture than
    by blocking here.

    Pass max_retries=0 for a single attempt with no retries -- e.g. an eval
    run against a tightly-capped daily quota, where retrying into an
    already-exhausted cap only burns requests that can't succeed anyway.
    """
    for attempt in range(max_retries + 1):
        try:
            return client.interactions.create(**kwargs)
        except RateLimitError as error:
            if attempt == max_retries:
                raise
            delay = _parse_retry_delay(error)
            if delay is None:
                delay = (2 ** (attempt + 1)) + random.uniform(0, 1)
            time.sleep(min(delay, _MAX_BACKOFF_SECONDS))


async def _call_tool_via_mcp(name: str, arguments: dict) -> dict:
    """Call one tool on the standalone MCP server (mcp_server/server.py) over
    Streamable HTTP and return its result as a plain dict.

    Connects fresh for this single call rather than reusing a session across
    a whole `ask()` invocation -- see the module docstring's design notes.
    The tool's return value comes back JSON-encoded in the first text content
    block (confirmed empirically: structured_content is not populated for
    these tools despite their dict return annotations), so it's decoded with
    json.loads rather than read from structured_content.
    """
    async with streamable_http_client(MCP_SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            call_result = await session.call_tool(name, arguments)
            if call_result.is_error:
                raise RuntimeError(
                    f"MCP tool {name!r} returned an error: {call_result.content}"
                )
            return json.loads(call_result.content[0].text)


def _unwrap_transport_error(error: BaseException) -> httpx2.TransportError | None:
    """Find an httpx2.TransportError inside a (possibly nested) ExceptionGroup.

    anyio's TaskGroup wraps connection failures in an ExceptionGroup rather
    than letting them propagate directly -- confirmed empirically against an
    unreachable MCP server. Returns the underlying transport error if one is
    found anywhere in the group, else None. None means "this isn't a
    connectivity failure" -- callers re-raise the original error as-is
    rather than mislabeling an unrelated bug as "server unreachable".
    """
    if isinstance(error, httpx2.TransportError):
        return error
    if isinstance(error, ExceptionGroup):
        for sub_error in error.exceptions:
            found = _unwrap_transport_error(sub_error)
            if found is not None:
                return found
    return None


def _execute_tool(name: str, arguments: dict, use_mcp: bool):
    """Execute one tool call, either directly (default) or via the MCP server.

    The direct path (use_mcp=False, the default) is exactly today's
    behavior: a plain registry lookup and call -- unchanged for /chat,
    ask_with_fallback, the eval harness, and every existing test. The MCP
    path (use_mcp=True) round-trips the same name/arguments through
    mcp_server/server.py instead.

    Only connectivity failures (an unreachable MCP server) are translated
    into a clear RuntimeError here -- there's no fixture fallback for this
    path, so that's meant to fail loudly, not degrade gracefully. Anything
    else -- a bug in _call_tool_via_mcp, a malformed response, an is_error
    result -- propagates as itself rather than being swallowed under a
    misleading "server unreachable" message.
    """
    if not use_mcp:
        return _TOOL_REGISTRY[name](**arguments)

    try:
        return asyncio.run(_call_tool_via_mcp(name, arguments))
    except (httpx2.TransportError, ExceptionGroup) as error:
        transport_error = _unwrap_transport_error(error)
        if transport_error is None:
            raise
        raise RuntimeError(
            f"MCP server unreachable at {MCP_SERVER_URL}: {transport_error}"
        ) from transport_error


def ask(
    question: str,
    max_iterations: int = 5,
    max_retries: int = _MAX_RETRIES,
    use_mcp: bool = False,
) -> dict:
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
        max_retries: Retries per LLM call on a 429 before raising, passed
            through to `_create_interaction`. Defaults to the same bounded
            backoff `/chat` and `ask_with_fallback` rely on; pass 0 to fail
            fast on the first rate limit instead (e.g. for eval runs).
        use_mcp: If True, execute tool calls via the standalone MCP server
            (mcp_server/server.py, see MCP_SERVER_URL in app/config.py)
            instead of calling app/tools.py directly. Defaults to False --
            every existing caller (/chat, ask_with_fallback, the eval
            harness) is on the direct path and unaffected by this option.

    Returns:
        {question, tool_calls: [{name, arguments, result, iteration}, ...],
        answer, iterations}. If the model still wants more tool calls after
        max_iterations rounds, `answer` is None and an `error` key explains
        the loop was stopped.
    """
    client = _get_client()
    history: list[dict] = [
        {"type": "user_input", "content": [{"type": "text", "text": question}]}
    ]
    all_tool_calls = []

    for iteration in range(1, max_iterations + 1):
        interaction = _create_interaction(
            client,
            max_retries=max_retries,
            model=MODEL,
            system_instruction=SYSTEM_PREAMBLE,
            input=history,
            tools=TOOL_SCHEMAS,
            store=False,
        )

        for step in interaction.steps:
            history.append(step.model_dump())

        function_calls = [s for s in interaction.steps if s.type == "function_call"]

        if not function_calls:
            return {
                "question": question,
                "tool_calls": all_tool_calls,
                "answer": interaction.output_text or None,
                "iterations": iteration,
            }

        for step in function_calls:
            result = _execute_tool(step.name, step.arguments, use_mcp)
            all_tool_calls.append(
                {
                    "name": step.name,
                    "arguments": step.arguments,
                    "result": result,
                    "iteration": iteration,
                }
            )
            history.append(
                {
                    "type": "function_result",
                    "name": step.name,
                    "call_id": step.id,
                    "result": [{"type": "text", "text": json.dumps(result)}],
                }
            )

    return {
        "question": question,
        "tool_calls": all_tool_calls,
        "answer": None,
        "error": f"stopped after {max_iterations} iterations without a final answer",
        "iterations": max_iterations,
    }


def _load_fixtures() -> dict[str, dict]:
    """Lazily load and cache fixtures/demo_fixtures.json, keyed by normalized question.

    See scripts/capture_fixtures.py for how this file is generated.
    """
    global _fixtures_cache
    if _fixtures_cache is None:
        if _FIXTURES_PATH.exists():
            _fixtures_cache = json.loads(_FIXTURES_PATH.read_text(encoding="utf-8"))
        else:
            _fixtures_cache = {}
    return _fixtures_cache


def ask_with_fallback(question: str, max_iterations: int = 5) -> dict:
    """Answer live via `ask`, falling back to a cached demo fixture if it can't.

    Catches only the specific failure modes a live Gemini call can hit that
    aren't the caller's fault -- a 429 that outlasted `_create_interaction`'s
    retries, a missing or rejected API key, or a connection failure -- and
    are worth degrading gracefully for rather than taking the whole endpoint
    down. Anything else (a bug in our own request, an unexpected exception)
    propagates so it's visible instead of silently serving stale fixture
    data.

    Adds a `demo_mode` key to the result: False for a live answer, True when
    served from fixtures/demo_fixtures.json. If the live call fails and the
    question isn't in the fixtures, returns a clean error dict (matching
    `ask`'s error shape) instead of raising, so /chat never crashes because
    Gemini is unavailable.
    """
    try:
        result = ask(question, max_iterations)
        result["demo_mode"] = False
        return result
    except (RuntimeError, RateLimitError, AuthenticationError, APIConnectionError) as error:
        cached = _load_fixtures().get(question.strip().lower())
        if cached is not None:
            return {**cached, "demo_mode": True}
        return {
            "question": question,
            "tool_calls": [],
            "answer": None,
            "error": f"live call failed and no demo fixture available: {error}",
            "iterations": 0,
            "demo_mode": False,
        }
