from __future__ import annotations

import pytest

from local_code.agent import DECLINED, Agent, AgentConfig
from local_code.session import Session
from tests.conftest import FakeClient, text_chunks, tool_call_chunks


def make_agent(client, yolo=False, confirm=None, max_iterations=25, notify=None):
    cfg = AgentConfig(model="m", max_iterations=max_iterations, yolo=yolo)
    session = Session(system_prompt="base")
    return Agent(client, session, cfg, use_native=True, confirm=confirm, notify=notify)


def test_final_answer_without_tools():
    client = FakeClient([text_chunks("hola!")])
    agent = make_agent(client)
    assert agent.run_turn("hola") == "hola!"
    assert agent.session.history == [
        {"role": "user", "content": "hola"},
        {"role": "assistant", "content": "hola!"},
    ]
    model, messages, tools = client.calls[0]
    assert tools is not None and len(tools) == 7
    assert messages[0]["role"] == "system"


def test_tool_call_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "f.txt").write_text("secreto")
    client = FakeClient([
        tool_call_chunks("read_file", {"path": "f.txt"}),
        text_chunks("el archivo dice secreto"),
    ])
    agent = make_agent(client)
    out = agent.run_turn("lee f.txt")
    assert out == "el archivo dice secreto"
    roles = [m["role"] for m in agent.session.history]
    assert roles == ["user", "assistant", "tool", "assistant"]
    tool_msg = agent.session.history[2]
    assert tool_msg == {"role": "tool", "tool_name": "read_file", "content": "secreto"}
    assert len(client.calls) == 2


def test_multiple_tool_calls_in_one_response(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("A")
    (tmp_path / "b.txt").write_text("B")
    both = [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "read_file", "arguments": {"path": "a.txt"}}},
                    {"function": {"name": "read_file", "arguments": {"path": "b.txt"}}},
                ],
            },
            "done": False,
        },
        {"message": {"role": "assistant", "content": ""}, "done": True},
    ]
    client = FakeClient([both, text_chunks("listo")])
    agent = make_agent(client)
    agent.run_turn("lee ambos")
    tool_msgs = [m for m in agent.session.history if m["role"] == "tool"]
    assert [m["content"] for m in tool_msgs] == ["A", "B"]


def test_confirmation_denied_blocks_tool(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([
        tool_call_chunks("write_file", {"path": "x.txt", "content": "data"}),
        text_chunks("ok no lo escribo"),
    ])
    agent = make_agent(client, confirm=lambda name, preview: False)
    agent.run_turn("escribí x.txt")
    assert not (tmp_path / "x.txt").exists()
    tool_msg = [m for m in agent.session.history if m["role"] == "tool"][0]
    assert tool_msg["content"] == DECLINED


def test_yolo_skips_confirmation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def explode(name, preview):
        raise AssertionError("confirm must not be called in yolo mode")

    client = FakeClient([
        tool_call_chunks("write_file", {"path": "x.txt", "content": "data"}),
        text_chunks("escrito"),
    ])
    agent = make_agent(client, yolo=True, confirm=explode)
    agent.run_turn("escribí x.txt")
    assert (tmp_path / "x.txt").read_text() == "data"


def test_readonly_tools_skip_confirmation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "f.txt").write_text("x")

    def explode(name, preview):
        raise AssertionError("confirm must not be called for read-only tools")

    client = FakeClient([
        tool_call_chunks("read_file", {"path": "f.txt"}),
        text_chunks("ok"),
    ])
    agent = make_agent(client, confirm=explode)
    agent.run_turn("lee")


def test_max_iterations_stops(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "f.txt").write_text("x")
    notes = []
    client = FakeClient([tool_call_chunks("read_file", {"path": "f.txt"})] * 3)
    agent = make_agent(client, max_iterations=3, notify=notes.append)
    agent.run_turn("loop")
    assert len(client.calls) == 3
    assert any("max iterations" in n.lower() for n in notes)


def test_on_token_streams():
    tokens = []
    client = FakeClient([text_chunks("hola!")])
    cfg = AgentConfig(model="m")
    agent = Agent(client, Session(), cfg, use_native=True, on_token=tokens.append)
    agent.run_turn("hola")
    assert "".join(tokens) == "hola!"
