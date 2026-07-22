from __future__ import annotations

import pytest

from local_code.agent import Agent, AgentConfig
from local_code.client import OllamaError
from local_code.session import Session
from tests.conftest import FakeClient, text_chunks, tool_call_chunks


def make_agent(client, events, use_native=True):
    return Agent(
        client,
        Session(system_prompt="base"),
        AgentConfig(model="m"),
        use_native=use_native,
        on_token=lambda t: events.append("tok"),
        on_stream_start=lambda: events.append("start"),
        on_stream_end=lambda: events.append("end"),
    )


def test_hooks_wrap_single_stream():
    events: list[str] = []
    agent = make_agent(FakeClient([text_chunks("hola")]), events)
    agent.run_turn("hi")
    assert events == ["start", "tok", "end"]


def test_hooks_once_per_model_call_native(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "f.txt").write_text("x")
    events: list[str] = []
    client = FakeClient([
        tool_call_chunks("read_file", {"path": "f.txt"}),
        text_chunks("listo"),
    ])
    agent = make_agent(client, events)
    agent.run_turn("lee")
    assert events.count("start") == 2
    assert events.count("end") == 2
    assert events[0] == "start" and events[-1] == "end"


def test_hooks_in_react_mode():
    events: list[str] = []
    agent = make_agent(FakeClient([text_chunks("respuesta")]), events, use_native=False)
    agent.run_turn("hola")
    assert events == ["start", "tok", "end"]


def test_stream_end_called_on_error():
    def exploding():
        yield {"message": {"role": "assistant", "content": "x"}, "done": False}
        raise OllamaError("boom")

    events: list[str] = []
    agent = make_agent(FakeClient([exploding()]), events)
    with pytest.raises(OllamaError):
        agent.run_turn("hola")
    assert events[-1] == "end"


def test_hooks_default_noop():
    agent = Agent(
        FakeClient([text_chunks("ok")]),
        Session(),
        AgentConfig(model="m"),
        use_native=True,
    )
    assert agent.run_turn("hola") == "ok"
