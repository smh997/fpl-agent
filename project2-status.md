# Project 2 Status — FPL Tool-Calling Agent

**STATUS: Stage 4 (`/chat` endpoint + tests) complete.** `app/main.py` wraps `agent.ask()` in a FastAPI `POST /chat`; `tests/test_chat.py` (4 tests, `ask` mocked) covers the HTTP contract; a live `uvicorn` run confirmed a real Cohere-backed answer end-to-end through the endpoint. Stage 5 (README + demo + push) not yet started.

## Goal

A tool-calling agent that answers Fantasy Premier League questions in natural language by planning and calling real API tools in a loop. Demonstrates: function/tool calling, the agent loop, and multi-step planning — the core of Module 7 (Tool Use), which is complete.

## Why FPL API

- Free, no auth on public endpoints.
- Base URL: `https://fantasy.premierleague.com/api/`
- Multiple distinct endpoints map naturally to separate tools → forces real multi-step reasoning.
- CORS is irrelevant: FastAPI calls it server-side.
- Caveat: unofficial/undocumented API, endpoints can change. It's July, so data reflects the completed season + new fixtures until August.

## Architecture (planned)

```
question → Cohere command-a-03-2025 (tool calling)
         → model returns tool_calls
         → backend executes tools against FPL API
         → results fed back → model calls more tools OR answers
         → loop until answer (max ~5 iterations)
```

Fresh repo `fpl-agent`, FastAPI backend, `/chat` endpoint runs the agent loop.

## Tool set (4 tools — composable primitives, not answer-givers)

| Tool | Input | Returns |
|---|---|---|
| `find_player` | name | `{id, name, team, position, price}` |
| `get_player_stats` | player_id | `{name, form, total_points, goals, assists, minutes, ppg}` |
| `get_team_fixtures` | team_id, next_n | list of `{opponent, venue, difficulty}` |
| `get_gameweek_summary` | gameweek (opt) | `{gameweek, avg_score, highest, most_captained, top_player}` |

Deliberately NO `compare_players` tool — comparison is the reasoning the agent should do itself by calling `get_player_stats` twice. Building it as a tool would steal the interesting part. Confirmed working in Stage 3: "Compare Haaland and Saka's points" made 2 `find_player` calls in one round, then 2 `get_player_stats` calls in the next.

## Gotchas to watch (from planning)

- **Tool schemas ARE prompts** — the model picks tools from `description` fields. Vague descriptions → wrong tool.
- **Cap the loop** at ~5 iterations — models can loop forever.
- **`bootstrap-static` is huge (~1-2 MB)** — tools MUST filter to small payloads, never return raw API blobs (context + cost).
- **Fuzzy name matching** — "Salah" → "Mohamed Salah". `find_player` needs forgiving matching or the agent dead-ends on step one.
- **Cache `bootstrap-static`** — one call backs several tools; fetch once, cache in memory.
- **Fuzzy matching needs a real threshold, not just "closest wins"** — pure `SequenceMatcher` ratio on short names gives coincidentally high scores for unrelated players (e.g. "Salah" vs "Saliba" scored 0.73), so "pick whatever's closest" silently returns the wrong person instead of admitting no good match. Fixed in `tools.py` with a substring-match fast path plus a typo-ratio floor of 0.78 for the fuzzy fallback.
- **Mohamed Salah is not in the live dataset as of this snapshot** (2026-07, pre-season) — useful smoke-test case for confirming `find_player` returns `{"error": "no player found"}` instead of a bogus match. Smoke script uses Haaland for the full happy-path chain instead.
- **Cohere tool-result content can't have a bare top-level `id` key** — the v2 chat API auto-parses tool message JSON content and treats a top-level `id` as a document id, which must be a string. `find_player`'s result has `id` as an int (the player id) and got rejected with `"A tool result's output's id field must be a string"`. Fixed in `agent.py` by wrapping every tool result as `{"result": <actual dict>}` before serializing into the tool message content, so no tool payload's own keys are ever read as protocol-level fields.
- **No system prompt = no grounding.** `ask()` originally sent only the bare user question with no system message. For open-ended questions like "Who is Haaland?", the model answered from its own training knowledge (`tool_calls: []`, a full bio) instead of calling `find_player` — nothing told it that was forbidden, and `find_player`'s old description ("get their id, team, position, and price") didn't semantically read as the answer to a "who is" question either. Fixed with a `SYSTEM_PREAMBLE` in `agent.py` (explicitly forbids answering from memory, requires tool use for any player/team/fixture/gameweek question, tells it to say so briefly if the tools can't answer) plus a broadened `find_player` description in `schemas.py` framing it as the required first step for *any* player question, not just numeric lookups. Verified live: "Who is Haaland?" now calls `find_player` and grounds the answer in its result; genuinely out-of-scope questions (e.g. "capital of France") get a brief decline instead of a free-associated answer.
- **`find_player` exposed team *name* but no team *id*, so the agent guessed and got the wrong number.** Chaining into `get_team_fixtures(team_id=...)` after `find_player` failed live with `{"error": "team not found"}` — the model passed `team_id=43` for Man City, which is actually Man City's `code` field (43), not its `id` (15). `find_player` resolved `best_element["team"]` (already the correct 1-20 id) all the way down to a name string and discarded the number, so the model had nothing grounded to chain with and fell back on memory, landing on the code. Fixed by adding `"team_id": best_element["team"]` to `find_player`'s return (kept `"team"` name too) and noting in its schema description to use `team_id`, not the name, for `get_team_fixtures`. `get_team_fixtures` itself needed no change — it was already keyed correctly by `team["id"]`. Regression-guarded in `tests/test_tools.py` (asserts `team_id` is in 1-20 and that it round-trips successfully into `get_team_fixtures`).

## Build stages

1. ✅ **Data layer** (fpl_client + 4 tool functions, tested against live API)
2. ✅ Tool schemas + Cohere tool-calling for a single call
3. ✅ The agent loop (multi-tool, iterate until answer)
4. ✅ `/chat` endpoint + tests
5. README + demo + push ← NEXT

## Next

Stage 5: write a README (setup, `.env` config, how to run `uvicorn app.main:app`, example `/chat` requests, architecture summary linking back to this doc) and push the repo. No code changes expected — this is documentation + `git init`/commit/push (repo is not yet a git repository).
