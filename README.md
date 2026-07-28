# FPL Agent

A tool-calling agent that answers Fantasy Premier League questions in natural language by planning and calling real FPL API tools in a loop.

**What this demonstrates**: tool/function calling, the agent loop, multi-step planning, and grounded answers (the model answers from live tool data, never from its own training memory).

## Architecture

```
Streamlit frontend
        |
        | POST /chat
        v
FastAPI backend
        |
        v
Agent loop <-> Cohere command-a-03-2025
        |
        | executes tool calls
        v
FPL API
```

The frontend sends each question to FastAPI and renders the returned answer and
tool-call trace. A system preamble instructs the model that it is an FPL data
assistant, that it must use the tools for any question about players, teams,
fixtures, or gameweeks, and that it must not answer such questions from its own
memory. The loop lives in `app/agent.py`'s `ask()`; each round executes whatever
tool calls the model requests against the FPL API and feeds the results back
until the model either answers or the iteration cap is hit.

## Tools

| Tool | Input | Returns |
|---|---|---|
| `find_player` | `name` | `{id, name, team, team_id, position, price}` |
| `get_player_stats` | `player_id` | `{name, form, total_points, goals_scored, assists, minutes, points_per_game}` |
| `get_team_fixtures` | `team_id`, `next_n` | `{team, fixtures: [{opponent, venue, difficulty}]}` |
| `get_gameweek_summary` | `gameweek` (optional) | `{gameweek, average_score, highest_score, most_captained_id, top_element_id}` |

There's deliberately no `compare_players` tool — comparison is reasoning the agent does itself, by calling `get_player_stats` twice and reasoning over both results.

## Tech stack

FastAPI, Cohere (`command-a-03-2025`, tool calling), httpx, pytest.

## Quickstart

Requires Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

cp .env.example .env
# edit .env and set COHERE_API_KEY (free key at https://dashboard.cohere.com)

uvicorn app.main:app --reload
```

Then either:
- `POST http://localhost:8000/chat` with a JSON body `{"question": "..."}`, or
- open `http://localhost:8000/docs` for the interactive Swagger UI.

## Frontend

The Streamlit frontend provides a chat UI for the FastAPI `/chat` endpoint. It
keeps conversation history across Streamlit reruns and places a **How the agent
reasoned** trace panel beneath each answer. Expanding the panel shows every tool
call in order, including its iteration, tool name, arguments, result, and the
overall iteration count.

![Frontend](docs/frontend.gif)

Terminal 1 — start the FastAPI backend:

```bash
uvicorn app.main:app --reload
```

Terminal 2 — start the Streamlit frontend:

```bash
streamlit run frontend/app.py
```

The frontend targets `http://localhost:8000` by default. Set the `BACKEND_URL`
environment variable to override that backend target:

```bash
# macOS/Linux
BACKEND_URL=http://localhost:9000 streamlit run frontend/app.py

# Windows PowerShell
$env:BACKEND_URL="http://localhost:9000"
streamlit run frontend/app.py
```

## Usage

Three real `/chat` interactions, showing tool grounding, multi-step chaining, and a committed recommendation.

### a) Grounded answer, not memory

`POST /chat`
```json
{"question": "Who is Haaland?"}
```

Response:
```json
{
  "question": "Who is Haaland?",
  "tool_calls": [
    {
      "name": "find_player",
      "arguments": {
        "name": "Haaland"
      },
      "result": {
        "id": 411,
        "name": "Haaland",
        "team": "Man City",
        "team_id": 15,
        "position": "FWD",
        "price": 15.5
      },
      "iteration": 1
    }
  ],
  "answer": "Haaland is a forward for Man City and costs £15.5.",
  "iterations": 2
}
```

### b) Two-step chain (comparison with no dedicated tool)

`POST /chat`
```json
{"question": "Compare Haaland and Saka's points"}
```

Response:
```json
{
  "question": "Compare Haaland and Saka's points",
  "tool_calls": [
    {
      "name": "find_player",
      "arguments": {
        "name": "Haaland"
      },
      "result": {
        "id": 411,
        "name": "Haaland",
        "team": "Man City",
        "team_id": 15,
        "position": "FWD",
        "price": 15.5
      },
      "iteration": 1
    },
    {
      "name": "find_player",
      "arguments": {
        "name": "Saka"
      },
      "result": {
        "id": 12,
        "name": "Saka",
        "team": "Arsenal",
        "team_id": 1,
        "position": "MID",
        "price": 9.5
      },
      "iteration": 1
    },
    {
      "name": "get_player_stats",
      "arguments": {
        "player_id": 411
      },
      "result": {
        "name": "Haaland",
        "form": 0,
        "total_points": 239,
        "goals_scored": 27,
        "assists": 8,
        "minutes": 2953,
        "points_per_game": 6.8
      },
      "iteration": 2
    },
    {
      "name": "get_player_stats",
      "arguments": {
        "player_id": 12
      },
      "result": {
        "name": "Saka",
        "form": 0,
        "total_points": 157,
        "goals_scored": 7,
        "assists": 10,
        "minutes": 2218,
        "points_per_game": 5.1
      },
      "iteration": 2
    }
  ],
  "answer": "Haaland has 239 points and Saka has 157 points.",
  "iterations": 3
}
```

### c) Three-tool chain with a committed recommendation

`POST /chat`
```json
{"question": "Is Haaland worth captaining based on his form and fixtures?"}
```

Response:
```json
{
  "question": "Is Haaland worth captaining based on his form and fixtures?",
  "tool_calls": [
    {
      "name": "find_player",
      "arguments": {
        "name": "Haaland"
      },
      "result": {
        "id": 411,
        "name": "Haaland",
        "team": "Man City",
        "team_id": 15,
        "position": "FWD",
        "price": 15.5
      },
      "iteration": 1
    },
    {
      "name": "get_player_stats",
      "arguments": {
        "player_id": 411
      },
      "result": {
        "name": "Haaland",
        "form": 0,
        "total_points": 239,
        "goals_scored": 27,
        "assists": 8,
        "minutes": 2953,
        "points_per_game": 6.8
      },
      "iteration": 2
    },
    {
      "name": "get_team_fixtures",
      "arguments": {
        "team_id": 15
      },
      "result": {
        "team": "Man City",
        "fixtures": [
          {
            "opponent": "Bournemouth",
            "venue": "home",
            "difficulty": 3
          },
          {
            "opponent": "Crystal Palace",
            "venue": "away",
            "difficulty": 3
          },
          {
            "opponent": "Coventry City",
            "venue": "home",
            "difficulty": 2
          },
          {
            "opponent": "Man Utd",
            "venue": "away",
            "difficulty": 4
          },
          {
            "opponent": "Sunderland",
            "venue": "home",
            "difficulty": 2
          }
        ]
      },
      "iteration": 2
    }
  ],
  "answer": "Haaland has scored 27 goals and assisted 8 times this season, with a points per game of 6.8. However, his form is 0. His next fixtures are against Bournemouth (home), Crystal Palace (away), Coventry City (home), Man Utd (away), and Sunderland (home), with difficulty ratings of 3, 3, 2, 4, and 2 respectively.\n\nBased on his form, Haaland is not worth captaining.",
  "iterations": 3
}
```

## Engineering notes

**Grounding required an explicit system preamble.** Early on, "Who is Haaland?" returned `tool_calls: []` and a full biography pulled from the model's training data — nothing told it that was off-limits. Adding a system preamble that states the model's training data is stale and it must use the tools (never memory) for any player/team/fixture/gameweek question fixed it; the `find_player` schema description was also broadened, since its old wording ("get id, team, position, price") didn't semantically match an open-ended "who is" question.

**`find_player` exposed the team name but not its id, so the model guessed.** Chaining into `get_team_fixtures` after `find_player` failed live with `{"error": "team not found"}` — the model passed team *code* 43 for Man City instead of team *id* 15, because `find_player` resolved the team all the way down to a name string and never surfaced the numeric id it already had internally. Fixed by adding `team_id` to `find_player`'s return alongside the name.

**Fuzzy name matching needed a real threshold.** A naive "closest match wins" approach using `SequenceMatcher` ratios scored "Salah" against "Saliba" at 0.73 — high enough to confidently return the wrong player instead of admitting no good match existed. Fixed with a substring-match fast path plus a stricter ratio floor for the pure-fuzzy fallback, so coincidental character overlap can no longer masquerade as a real match.

**Cohere's v2 chat API rejected a tool result's own `id` field.** Tool results are round-tripped back to the model as JSON text, and the API auto-parses that JSON, treating a top-level `id` key as a document id that must be a string. `find_player`'s result legitimately has an integer `id` (the player id), which collided and got rejected. Fixed by wrapping every tool result as `{"result": {...}}` before serializing it into the tool message, so none of the tools' own field names are ever read as protocol-level fields.

## Known limitations

- Recommendations (e.g. captaincy judgments) weight recent `form` heavily. Since this is currently pre-season (July), `form` is `0` for every player, which skews those recommendations toward season-long totals by default rather than genuine recent form — worth re-checking once the season starts and form data populates.
- This uses an unofficial, undocumented FPL API (`fantasy.premierleague.com/api/`) with no stability guarantees; endpoints or field names can change without notice.
- Data reflects the completed prior season plus new fixtures until the new season kicks off.

## Future work

- Weight season-long metrics more heavily when `form` data is sparse or zero (e.g. pre-season).
- Deployment (containerization, hosting).
