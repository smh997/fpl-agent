"""Tests for the opt-in MCP tool-execution path (ask(..., use_mcp=True)).

Mocks the MCP client boundary -- app.agent.ClientSession and
app.agent.streamable_http_client -- the same way test_chat.py monkeypatches
app.agent.ask: no real MCP server, network, or Gemini call is ever made.
"""

import json

import httpx2
import pytest

from app import agent


class _FakeTextContent:
    def __init__(self, text):
        self.text = text


class _FakeCallToolResult:
    def __init__(self, content, is_error=False):
        self.content = content
        self.is_error = is_error


def _fake_call_result(payload: dict) -> _FakeCallToolResult:
    """A CallToolResult matching the real shape confirmed against the live
    server: the tool's dict return value JSON-encoded in the first text
    content block (structured_content is not populated for these tools)."""
    return _FakeCallToolResult(content=[_FakeTextContent(json.dumps(payload))])


def _install_fake_mcp(monkeypatch, *, result=None, raise_error=None):
    """Patch agent.ClientSession / agent.streamable_http_client so call_tool()
    either returns `result` or raises `raise_error`, with no real network or
    MCP server involved.
    """

    class _FakeSession:
        def __init__(self, read, write):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def initialize(self):
            pass

        async def call_tool(self, name, arguments):
            if raise_error is not None:
                raise raise_error
            return result

    class _FakeTransport:
        def __call__(self, url):
            return self

        async def __aenter__(self):
            return (None, None)

        async def __aexit__(self, *exc_info):
            return False

    monkeypatch.setattr(agent, "ClientSession", _FakeSession)
    monkeypatch.setattr(agent, "streamable_http_client", _FakeTransport())


def test_execute_tool_direct_path_uses_registry(monkeypatch):
    """use_mcp=False must still be the plain registry call -- unaffected by
    anything added for the MCP path."""
    monkeypatch.setitem(
        agent._TOOL_REGISTRY, "find_player", lambda name: {"id": 411, "name": name}
    )

    result = agent._execute_tool("find_player", {"name": "Haaland"}, False)

    assert result == {"id": 411, "name": "Haaland"}


def test_execute_tool_via_mcp_returns_decoded_dict(monkeypatch):
    """content[0].text is JSON-encoded -- _execute_tool must decode it back
    into the same plain dict shape the direct path returns."""
    expected = {
        "id": 411,
        "name": "Haaland",
        "team": "Man City",
        "team_id": 15,
        "position": "FWD",
        "price": 15.5,
    }
    _install_fake_mcp(monkeypatch, result=_fake_call_result(expected))

    result = agent._execute_tool("find_player", {"name": "Haaland"}, True)

    assert result == expected


def test_mcp_path_matches_direct_path_shape(monkeypatch):
    """The whole use_mcp design rests on this: routing through MCP must
    produce the identical dict the direct path returns for the same tool."""
    expected = {
        "id": 411,
        "name": "Haaland",
        "team": "Man City",
        "team_id": 15,
        "position": "FWD",
        "price": 15.5,
    }
    monkeypatch.setitem(agent._TOOL_REGISTRY, "find_player", lambda name: expected)
    direct_result = agent._execute_tool("find_player", {"name": "Haaland"}, False)

    _install_fake_mcp(monkeypatch, result=_fake_call_result(expected))
    mcp_result = agent._execute_tool("find_player", {"name": "Haaland"}, True)

    assert mcp_result == direct_result
    assert mcp_result.keys() == direct_result.keys()


def test_execute_tool_via_mcp_raises_on_is_error(monkeypatch):
    """is_error=True must fail loudly instead of attempting json.loads on a
    payload that may not match the expected shape."""
    _install_fake_mcp(
        monkeypatch,
        result=_FakeCallToolResult(content=["no player found"], is_error=True),
    )

    with pytest.raises(RuntimeError, match="returned an error"):
        agent._execute_tool("find_player", {"name": "Nope"}, True)


def test_execute_tool_via_mcp_unreachable_raises_clear_runtime_error(monkeypatch):
    """A connection failure (wrapped in an anyio ExceptionGroup, confirmed
    empirically against a real unreachable server) must surface as a clear
    RuntimeError naming MCP_SERVER_URL and the actual underlying error --
    not a raw ExceptionGroup."""
    underlying = httpx2.ConnectError("All connection attempts failed")
    group = ExceptionGroup("unhandled errors in a TaskGroup", [underlying])
    _install_fake_mcp(monkeypatch, raise_error=group)

    with pytest.raises(RuntimeError) as exc_info:
        agent._execute_tool("find_player", {"name": "Haaland"}, True)

    assert agent.MCP_SERVER_URL in str(exc_info.value)
    assert "All connection attempts failed" in str(exc_info.value)


def test_execute_tool_via_mcp_unrelated_bug_propagates(monkeypatch):
    """A bug unrelated to connectivity (e.g. a KeyError) must propagate as
    itself, not get relabeled as 'server unreachable'."""
    _install_fake_mcp(monkeypatch, raise_error=KeyError("simulated bug"))

    with pytest.raises(KeyError):
        agent._execute_tool("find_player", {"name": "Haaland"}, True)
