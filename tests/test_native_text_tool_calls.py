from __future__ import annotations

import json

from local_code.agent import Agent, AgentConfig
from local_code.session import Session
from tests.conftest import FakeClient, text_chunks


def make_native_agent(client, **kwargs):
    cfg = AgentConfig(model="m", yolo=True, **kwargs)
    return Agent(client, Session(system_prompt="base"), cfg, use_native=True)


def test_native_executes_tool_call_leaked_as_bare_json(tmp_path, monkeypatch):
    """qwen2.5-coder and other local models sometimes emit a tool call as plain
    text (a bare JSON object) instead of a structured tool_calls array, even in
    native mode. The agent must still execute it."""
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "numbers.txt"
    leaked = json.dumps(
        {"name": "write_file", "arguments": {"path": str(target), "content": "1\n2\n3"}}
    )
    client = FakeClient([text_chunks(leaked), text_chunks("listo, archivo creado")])
    agent = make_native_agent(client)

    out = agent.run_turn("creá numbers.txt con 1 2 3")

    assert target.read_text() == "1\n2\n3"
    assert out == "listo, archivo creado"
    # a tool result must be in the history
    tool_msgs = [m for m in agent.session.history if m["role"] == "tool"]
    assert tool_msgs and tool_msgs[0]["tool_name"] == "write_file"


def test_native_executes_tagged_tool_call_in_text(tmp_path, monkeypatch):
    """Same, but the model wrapped the call in <tool_call> tags in the content."""
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "x.txt"
    tagged = (
        "Voy a crearlo.\n"
        '<tool_call>{"name": "write_file", "arguments": '
        f'{{"path": "{target}", "content": "hola"}}}}</tool_call>'
    )
    client = FakeClient([text_chunks(tagged), text_chunks("hecho")])
    agent = make_native_agent(client)

    agent.run_turn("creá x.txt")
    assert target.read_text() == "hola"


def test_native_does_not_execute_plain_prose(tmp_path, monkeypatch):
    """A normal answer that merely mentions JSON or an unknown name must NOT be
    treated as a tool call — only real registered tool names trigger execution."""
    monkeypatch.chdir(tmp_path)
    prose = 'El formato es {"name": "algo", "arguments": {}} pero no lo ejecutes.'
    client = FakeClient([text_chunks(prose)])
    agent = make_native_agent(client)

    out = agent.run_turn("explicame el formato")
    assert out == prose
    assert not any(m["role"] == "tool" for m in agent.session.history)
    assert len(client.calls) == 1  # no follow-up turn triggered
