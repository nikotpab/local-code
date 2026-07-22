from __future__ import annotations

from local_code.tools import bash
from local_code.tools.context import ToolContext


def test_bash_captures_stdout_and_exit_code():
    out = bash.run({"command": "echo hola"}, ToolContext())
    assert "exit code: 0" in out
    assert "hola" in out


def test_bash_captures_stderr_and_nonzero_exit():
    out = bash.run({"command": "echo oops >&2; exit 3"}, ToolContext())
    assert "exit code: 3" in out
    assert "oops" in out


def test_bash_timeout():
    out = bash.run({"command": "sleep 5"}, ToolContext(bash_timeout=1))
    assert out == "Error: command timed out after 1s"


def test_bash_truncates_output():
    out = bash.run({"command": "python3 -c \"print('x' * 20000)\""}, ToolContext())
    assert len(out) < 21000


def test_bash_preview():
    assert bash.preview({"command": "ls -la"}) == "$ ls -la"
    assert bash.REQUIRES_CONFIRMATION is True
