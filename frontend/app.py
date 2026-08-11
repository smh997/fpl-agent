"""Streamlit chat client for the FPL agent's FastAPI endpoint."""

import json
import os
from typing import Any

import requests
import streamlit as st


st.set_page_config(page_title="FPL Agent", page_icon="⚽", layout="centered")


def _resolve_backend_url() -> str:
    """Resolve the backend URL: Streamlit Cloud secrets, then env var, then local default."""
    try:
        if "BACKEND_URL" in st.secrets:
            return st.secrets["BACKEND_URL"]
    except Exception:
        pass  # no secrets.toml locally -- st.secrets can raise just for being accessed
    return os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


BACKEND_URL = _resolve_backend_url().rstrip("/")
CHAT_ENDPOINT = f"{BACKEND_URL}/chat"
# Generous timeout: Render's free tier sleeps after 15 min idle and can take
# up to ~50s to wake, so the first request after a period of inactivity is
# slow by design, not broken.
REQUEST_TIMEOUT = 90


def format_json(value: Any) -> str:
    """Return readable JSON for tool arguments and results."""
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def render_trace(tool_calls: list[dict[str, Any]], iterations: Any) -> None:
    """Render an ordered, legible view of the agent's tool-use chain."""
    with st.expander("How the agent reasoned"):
        iteration_label = iterations if iterations is not None else "unknown"
        st.caption(
            f"{len(tool_calls)} tool call{'s' if len(tool_calls) != 1 else ''} "
            f"across {iteration_label} iteration{'s' if iterations != 1 else ''}"
        )

        if not tool_calls:
            st.write("The agent answered without calling a tool.")
            return

        for index, call in enumerate(tool_calls, start=1):
            name = call.get("name", "unknown_tool")
            iteration = call.get("iteration", "?")
            st.markdown(f"**{index}. `{name}`** · Iteration {iteration}")

            arguments_column, result_column = st.columns(2)
            with arguments_column:
                st.caption("Arguments")
                st.code(format_json(call.get("arguments", {})), language="json")
            with result_column:
                st.caption("Result")
                st.code(format_json(call.get("result")), language="json")

            if index < len(tool_calls):
                st.divider()


def render_assistant_message(message: dict[str, Any]) -> None:
    """Render an assistant answer, error state, and optional reasoning trace."""
    if message.get("is_error"):
        st.error(message["content"])
    else:
        if message.get("demo_mode"):
            st.caption("🔄 cached demo response")
        st.markdown(message["content"])
        render_trace(
            message.get("tool_calls", []),
            message.get("iterations"),
        )


def friendly_http_error(response: requests.Response) -> str:
    """Build a concise error message from a non-success backend response."""
    try:
        body = response.json()
        detail = body.get("detail") if isinstance(body, dict) else None
    except ValueError:
        detail = None

    suffix = f" The backend says: {detail}" if detail else ""
    return (
        f"The FPL agent returned an error (HTTP {response.status_code})."
        f"{suffix} Please try again."
    )


def ask_backend(question: str) -> dict[str, Any]:
    """Send one question to the backend and return its JSON response."""
    try:
        response = requests.post(
            CHAT_ENDPOINT,
            json={"question": question},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.ConnectionError as exc:
        raise RuntimeError(
            "I couldn't reach the FPL agent. Make sure the backend is "
            f"running at {BACKEND_URL}."
        ) from exc
    except requests.Timeout as exc:
        raise RuntimeError(
            "The FPL agent took too long to respond, even after waiting for "
            "a possible cold start. Please try again."
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(
            "Something went wrong while contacting the FPL agent. Please try again."
        ) from exc

    if not response.ok:
        raise RuntimeError(friendly_http_error(response))

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "The FPL agent returned an unreadable response. Please try again."
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError(
            "The FPL agent returned an unexpected response. Please try again."
        )

    return payload


st.title("⚽ FPL Agent")
st.write("Ask a Fantasy Premier League question and inspect how the agent finds its answer.")
st.caption(
    "Try: “Compare Haaland and Saka's points” · "
    "“Is Salah worth captaining?” · "
    "“Show Arsenal's next five fixtures”"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for stored_message in st.session_state.messages:
    with st.chat_message(stored_message["role"]):
        if stored_message["role"] == "assistant":
            render_assistant_message(stored_message)
        else:
            st.markdown(stored_message["content"])

if question := st.chat_input("Ask about players, form, fixtures, or gameweeks"):
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(
            "Waking up the backend — the first request can take up to a "
            "minute on the free tier..."
        ):
            try:
                response_data = ask_backend(question)
                answer = response_data.get("answer")
                backend_error = response_data.get("error")

                if answer:
                    assistant_message = {
                        "role": "assistant",
                        "content": str(answer),
                        "tool_calls": response_data.get("tool_calls", []),
                        "iterations": response_data.get("iterations"),
                        "demo_mode": response_data.get("demo_mode", False),
                        "is_error": False,
                    }
                else:
                    assistant_message = {
                        "role": "assistant",
                        "content": str(
                            backend_error
                            or "The agent finished without returning an answer. "
                            "Please try rephrasing your question."
                        ),
                        "is_error": True,
                    }
            except RuntimeError as exc:
                assistant_message = {
                    "role": "assistant",
                    "content": str(exc),
                    "is_error": True,
                }

        st.session_state.messages.append(assistant_message)
        render_assistant_message(assistant_message)
