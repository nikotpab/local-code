from __future__ import annotations

from local_code.tools import set_todos
from local_code.tools.context import ToolContext


def test_contract():
    assert set_todos.NAME == "set_todos"
    assert set_todos.REQUIRES_CONFIRMATION is False
    assert set_todos.PARAMETERS["type"] == "object"


def test_valid_todos_call_on_todos():
    received = []
    ctx = ToolContext(on_todos=received.append)
    todos = [
        {"text": "leer archivo", "status": "done"},
        {"text": "editar", "status": "in_progress"},
        {"text": "testear", "status": "pending"},
    ]
    out = set_todos.run({"todos": todos}, ctx)
    assert out == "Todos updated (3 items)"
    assert received == [todos]


def test_no_on_todos_still_works():
    out = set_todos.run({"todos": [{"text": "x", "status": "pending"}]}, ToolContext())
    assert out == "Todos updated (1 items)"


def test_not_a_list():
    out = set_todos.run({"todos": "nope"}, ToolContext())
    assert out.startswith("Error:")


def test_bad_status():
    out = set_todos.run({"todos": [{"text": "x", "status": "later"}]}, ToolContext())
    assert out.startswith("Error: item 0")


def test_missing_text():
    out = set_todos.run({"todos": [{"status": "pending"}]}, ToolContext())
    assert out.startswith("Error: item 0")


def test_context_backward_compatible():
    ctx = ToolContext(bash_timeout=5)
    assert ctx.on_todos is None
