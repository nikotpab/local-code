from __future__ import annotations

import pytest

from local_code.agent import DECLINED, DEFAULT_SYSTEM_PROMPT, Agent, AgentConfig
from local_code.session import Session
from tests.conftest import FakeClient, text_chunks, tool_call_chunks


class FakeCompactor:
    def __init__(self):
        self.calls = 0
        self.history_len_at_call = None

    def maybe_compact(self, session):
        self.calls += 1
        self.history_len_at_call = len(session.history)
        return False


class FakePermissions:
    def __init__(self, allowed=False):
        self.allowed = allowed
        self.allow_calls = []

    def is_allowed(self, name, arguments):
        return self.allowed

    def allow(self, name, arguments):
        self.allow_calls.append((name, arguments))


def make_agent(client, **kwargs):
    cfg = AgentConfig(model="m")
    return Agent(client, Session(system_prompt="base"), cfg, use_native=True, **kwargs)


def test_compactor_called_before_user_message():
    compactor = FakeCompactor()
    agent = make_agent(FakeClient([text_chunks("ok")]), compactor=compactor)
    agent.run_turn("hola")
    assert compactor.calls == 1
    assert compactor.history_len_at_call == 0


def test_permission_store_allows_without_confirm(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def explode(name, preview):
        raise AssertionError("confirm must not be called when allowed")

    client = FakeClient([
        tool_call_chunks("write_file", {"path": "x.txt", "content": "d"}),
        text_chunks("listo"),
    ])
    agent = make_agent(client, permission_store=FakePermissions(allowed=True), confirm=explode)
    agent.run_turn("dale")
    assert (tmp_path / "x.txt").read_text() == "d"


def test_always_registers_and_executes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    perms = FakePermissions(allowed=False)
    client = FakeClient([
        tool_call_chunks("write_file", {"path": "x.txt", "content": "d"}),
        text_chunks("listo"),
    ])
    agent = make_agent(client, permission_store=perms, confirm=lambda n, p: "always")
    agent.run_turn("dale")
    assert (tmp_path / "x.txt").exists()
    assert perms.allow_calls == [("write_file", {"path": "x.txt", "content": "d"})]


def test_bool_confirm_still_works(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([
        tool_call_chunks("write_file", {"path": "x.txt", "content": "d"}),
        text_chunks("ok"),
    ])
    agent = make_agent(client, confirm=lambda n, p: False)
    agent.run_turn("dale")
    assert not (tmp_path / "x.txt").exists()
    tool_msg = [m for m in agent.session.history if m["role"] == "tool"][0]
    assert tool_msg["content"] == DECLINED


def test_on_todos_reaches_tool():
    received = []
    todos = [{"text": "paso 1", "status": "pending"}]
    client = FakeClient([
        tool_call_chunks("set_todos", {"todos": todos}),
        text_chunks("planificado"),
    ])
    agent = make_agent(client, on_todos=received.append)
    agent.run_turn("plan")
    assert received == [todos]


def test_system_prompt_mentions_set_todos():
    assert "set_todos" in DEFAULT_SYSTEM_PROMPT


def run_write_turn(tmp_path, monkeypatch, **kwargs):
    """Run one turn where the model asks to write x.txt, and report the outcome."""
    monkeypatch.chdir(tmp_path)
    client = FakeClient([
        tool_call_chunks("write_file", {"path": "x.txt", "content": "d"}),
        text_chunks("listo"),
    ])
    agent = make_agent(client, **kwargs)
    agent.run_turn("dale")
    tool_msg = [m for m in agent.session.history if m["role"] == "tool"][0]
    return (tmp_path / "x.txt").exists(), tool_msg["content"]


@pytest.mark.parametrize("bad_decision", [None, "", 0, [], "maybe", "ALWAYS"])
def test_unrecognized_confirmation_fails_closed(tmp_path, monkeypatch, bad_decision):
    written, message = run_write_turn(
        tmp_path, monkeypatch, confirm=lambda n, p: bad_decision
    )
    assert written is False
    assert message == DECLINED


def test_yes_string_executes(tmp_path, monkeypatch):
    written, _ = run_write_turn(tmp_path, monkeypatch, confirm=lambda n, p: "yes")
    assert written is True


class ExplodingPermissions:
    def __init__(self, raise_on_allow=False):
        self.raise_on_allow = raise_on_allow
        self.confirm_was_asked = False

    def is_allowed(self, name, arguments):
        if not self.raise_on_allow:
            raise RuntimeError("corrupt store")
        return False

    def allow(self, name, arguments):
        if self.raise_on_allow:
            raise RuntimeError("cannot persist")


def test_is_allowed_failure_falls_back_to_asking(tmp_path, monkeypatch):
    perms = ExplodingPermissions()

    def confirm(name, preview):
        perms.confirm_was_asked = True
        return "no"

    written, message = run_write_turn(
        tmp_path, monkeypatch, permission_store=perms, confirm=confirm
    )
    assert perms.confirm_was_asked is True
    assert written is False
    assert message == DECLINED


def test_allow_failure_does_not_abort_execution(tmp_path, monkeypatch):
    written, _ = run_write_turn(
        tmp_path,
        monkeypatch,
        permission_store=ExplodingPermissions(raise_on_allow=True),
        confirm=lambda n, p: "always",
    )
    assert written is True
