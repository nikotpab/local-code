from __future__ import annotations

from local_code.session import Session


def test_messages_include_system_first():
    s = Session(system_prompt="sos un agente")
    s.add({"role": "user", "content": "hola"})
    assert s.messages == [
        {"role": "system", "content": "sos un agente"},
        {"role": "user", "content": "hola"},
    ]


def test_no_system_prompt():
    s = Session()
    s.add({"role": "user", "content": "hola"})
    assert s.messages == [{"role": "user", "content": "hola"}]


def test_clear_keeps_system_prompt():
    s = Session(system_prompt="base")
    s.add({"role": "user", "content": "hola"})
    s.clear()
    assert s.history == []
    assert s.messages == [{"role": "system", "content": "base"}]


def test_add_preserves_extra_fields():
    s = Session()
    msg = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "x"}}]}
    s.add(msg)
    assert s.history[0]["tool_calls"] == [{"function": {"name": "x"}}]
