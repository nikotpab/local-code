from __future__ import annotations

import io

from rich.console import Console

from local_code.cli import (
    COMPLETION_FLAGS,
    COMPLETION_SHELLS,
    run_completion_command,
)


def _console():
    return Console(file=io.StringIO(), force_terminal=False)


def test_completion_no_shell_is_usage_error():
    assert run_completion_command([], _console()) == 2


def test_completion_unknown_shell():
    assert run_completion_command(["powershell"], _console()) == 2


def test_completion_all_shells_emit_script(capsys):
    for shell in COMPLETION_SHELLS:
        code = run_completion_command([shell], _console())
        assert code == 0
        out = capsys.readouterr().out
        assert "local-code" in out
        # every declared flag appears in the emitted script
        for flag in COMPLETION_FLAGS:
            assert flag in out
