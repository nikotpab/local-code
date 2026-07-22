from __future__ import annotations

from local_code.agent import Agent, AgentConfig
from local_code.session import Session
from tests.conftest import FakeClient, text_chunks


def make_react_agent(client, max_iterations=25, notify=None, confirm=None, yolo=False):
    cfg = AgentConfig(model="m", max_iterations=max_iterations, yolo=yolo)
    session = Session(system_prompt="base prompt")
    return Agent(
        client, session, cfg, use_native=False, notify=notify, confirm=confirm
    )


def tool_call_text(name: str, arguments_json: str) -> str:
    return f'<tool_call>{{"name": "{name}", "arguments": {arguments_json}}}</tool_call>'


def test_no_tools_param_and_augmented_system_prompt():
    client = FakeClient([text_chunks("hola")])
    agent = make_react_agent(client)
    agent.run_turn("hola")
    model, messages, tools_param = client.calls[0]
    assert tools_param is None
    assert messages[0]["role"] == "system"
    assert "base prompt" in messages[0]["content"]
    assert "<tool_call>" in messages[0]["content"]
    assert agent.session.system_prompt == "base prompt"


def test_happy_path_tool_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "f.txt").write_text("secreto")
    client = FakeClient([
        text_chunks("Leo el archivo.\n" + tool_call_text("read_file", '{"path": "f.txt"}')),
        text_chunks("dice secreto"),
    ])
    agent = make_react_agent(client)
    out = agent.run_turn("lee f.txt")
    assert out == "dice secreto"
    obs = agent.session.history[2]
    assert obs["role"] == "user"
    assert obs["content"] == "Observation: secreto"


def test_multiple_calls_single_observation_message(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("A")
    (tmp_path / "b.txt").write_text("B")
    two = (
        tool_call_text("read_file", '{"path": "a.txt"}')
        + "\n"
        + tool_call_text("read_file", '{"path": "b.txt"}')
    )
    client = FakeClient([text_chunks(two), text_chunks("listo")])
    agent = make_react_agent(client)
    agent.run_turn("lee ambos")
    obs = agent.session.history[2]
    assert obs["content"] == "Observation: A\n\nObservation: B"


def test_malformed_then_recovers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "f.txt").write_text("ok")
    client = FakeClient([
        text_chunks('<tool_call>{"name": broken}</tool_call>'),
        text_chunks(tool_call_text("read_file", '{"path": "f.txt"}')),
        text_chunks("final"),
    ])
    agent = make_react_agent(client)
    assert agent.run_turn("dale") == "final"
    error_obs = agent.session.history[2]
    assert error_obs["role"] == "user"
    assert "Invalid JSON" in error_obs["content"]


def test_three_consecutive_malformed_aborts():
    notes = []
    bad = text_chunks('<tool_call>{"name": broken}</tool_call>')
    client = FakeClient([bad, bad, bad])
    agent = make_react_agent(client, notify=notes.append)
    agent.run_turn("dale")
    assert len(client.calls) == 3
    assert any("Aborting" in n for n in notes)


def test_unknown_tool_counts_as_failure():
    client = FakeClient([
        text_chunks(tool_call_text("teleport", "{}")),
        text_chunks("final"),
    ])
    agent = make_react_agent(client)
    assert agent.run_turn("dale") == "final"
    obs = agent.session.history[2]
    assert "unknown tool 'teleport'" in obs["content"]


def test_tool_internal_error_does_not_abort(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    missing = text_chunks(tool_call_text("read_file", '{"path": "nope.txt"}'))
    client = FakeClient([missing, missing, missing, text_chunks("me rindo")])
    agent = make_react_agent(client)
    assert agent.run_turn("lee nope") == "me rindo"
    assert len(client.calls) == 4


def test_max_iterations_react(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "f.txt").write_text("x")
    notes = []
    call = text_chunks(tool_call_text("read_file", '{"path": "f.txt"}'))
    client = FakeClient([call, call, call])
    agent = make_react_agent(client, max_iterations=3, notify=notes.append)
    agent.run_turn("loop")
    assert len(client.calls) == 3
    assert any("max iterations" in n.lower() for n in notes)


def test_confirmation_denied_in_react(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([
        text_chunks(tool_call_text("write_file", '{"path": "x.txt", "content": "d"}')),
        text_chunks("ok"),
    ])
    agent = make_react_agent(client, confirm=lambda name, preview: False)
    agent.run_turn("escribí")
    assert not (tmp_path / "x.txt").exists()
    obs = agent.session.history[2]
    assert "declined" in obs["content"].lower()
