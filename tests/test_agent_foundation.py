from __future__ import annotations

from local_code.agent import Agent, AgentConfig
from local_code.hooks import HookResult
from local_code.session import Session
from tests.conftest import FakeClient, text_chunks, tool_call_chunks


class FakeHooks:
    def __init__(self, pre=None, post=""):
        self.pre = pre or HookResult(False, "")
        self.post = post
        self.pre_calls = []
        self.post_calls = []

    def run_pre_tool(self, tool_name, arguments):
        self.pre_calls.append((tool_name, arguments))
        return self.pre

    def run_post_tool(self, tool_name, arguments, result):
        self.post_calls.append((tool_name, arguments, result))
        return self.post


class FakeCheckpoints:
    def __init__(self):
        self.calls = []
        self.seen_content = []

    def snapshot(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        path = arguments.get("path")
        if isinstance(path, str):
            from pathlib import Path

            p = Path(path)
            self.seen_content.append(p.read_text() if p.is_file() else None)
        return None


def make_agent(client, **kwargs):
    cfg = AgentConfig(model="m", yolo=True)
    return Agent(client, Session(system_prompt="base"), cfg, use_native=True, **kwargs)


def write_turn(tmp_path, monkeypatch, **kwargs):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([
        tool_call_chunks("write_file", {"path": "x.txt", "content": "nuevo"}),
        text_chunks("listo"),
    ])
    agent = make_agent(client, **kwargs)
    agent.run_turn("dale")
    tool_msg = [m for m in agent.session.history if m["role"] == "tool"][0]
    return agent, tool_msg["content"]


def test_blocking_hook_prevents_execution(tmp_path, monkeypatch):
    hooks = FakeHooks(pre=HookResult(True, "no toques eso"))
    _, message = write_turn(tmp_path, monkeypatch, hook_runner=hooks)
    assert not (tmp_path / "x.txt").exists()
    assert message == "Blocked by hook: no toques eso"
    assert hooks.pre_calls[0][0] == "write_file"
    assert hooks.post_calls == []


def test_allowing_hook_lets_tool_run(tmp_path, monkeypatch):
    hooks = FakeHooks()
    _, message = write_turn(tmp_path, monkeypatch, hook_runner=hooks)
    assert (tmp_path / "x.txt").read_text() == "nuevo"
    assert message.startswith("Wrote")


def test_post_hook_output_is_notified(tmp_path, monkeypatch):
    notes = []
    hooks = FakeHooks(post="revisado por el linter")
    write_turn(tmp_path, monkeypatch, hook_runner=hooks, notify=notes.append)
    assert any("revisado por el linter" in n for n in notes)
    assert hooks.post_calls[0][0] == "write_file"


def test_checkpoint_taken_before_execution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "x.txt").write_text("viejo")
    checkpoints = FakeCheckpoints()
    client = FakeClient([
        tool_call_chunks("write_file", {"path": "x.txt", "content": "nuevo"}),
        text_chunks("listo"),
    ])
    make_agent(client, checkpoint_store=checkpoints).run_turn("dale")
    assert checkpoints.calls[0][0] == "write_file"
    assert checkpoints.seen_content == ["viejo"]
    assert (tmp_path / "x.txt").read_text() == "nuevo"


def test_blocked_tool_is_not_checkpointed(tmp_path, monkeypatch):
    checkpoints = FakeCheckpoints()
    write_turn(
        tmp_path,
        monkeypatch,
        hook_runner=FakeHooks(pre=HookResult(True, "no")),
        checkpoint_store=checkpoints,
    )
    assert checkpoints.calls == []


def test_defaults_unchanged(tmp_path, monkeypatch):
    _, message = write_turn(tmp_path, monkeypatch)
    assert (tmp_path / "x.txt").read_text() == "nuevo"
    assert message.startswith("Wrote")


def test_tool_callbacks_fire_in_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "f.txt").write_text("contenido")
    events = []
    client = FakeClient([
        tool_call_chunks("read_file", {"path": "f.txt"}),
        text_chunks("listo"),
    ])
    agent = make_agent(
        client,
        on_tool_start=lambda n, a: events.append(("start", n, a)),
        on_tool_end=lambda n, r: events.append(("end", n, r)),
    )
    agent.run_turn("leé f.txt")
    kinds = [e[0] for e in events]
    assert kinds == ["start", "end"]
    assert events[0][1] == "read_file"
    assert events[0][2] == {"path": "f.txt"}
    assert events[1][1] == "read_file"
    assert events[1][2] == "contenido"


def test_tool_callbacks_default_noop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "f.txt").write_text("x")
    client = FakeClient([
        tool_call_chunks("read_file", {"path": "f.txt"}),
        text_chunks("ok"),
    ])
    # No on_tool_start/on_tool_end passed — must not raise.
    assert make_agent(client).run_turn("leé") == "ok"
