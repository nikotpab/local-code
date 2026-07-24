from __future__ import annotations

import json
import stat

from local_code.hooks import HookResult, HookRunner


def write_hook(dir, name, body, executable=True):
    dir.mkdir(parents=True, exist_ok=True)
    path = dir / name
    path.write_text("#!/bin/sh\n" + body)
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def test_missing_dir_does_not_block(tmp_path):
    runner = HookRunner(dir=tmp_path / "nope")
    assert runner.run_pre_tool("bash", {"command": "ls"}) == HookResult(False, "")


def test_missing_hook_does_not_block(tmp_path):
    (tmp_path / "hooks").mkdir()
    runner = HookRunner(dir=tmp_path / "hooks")
    assert runner.run_pre_tool("bash", {}).blocked is False


def test_exit_zero_allows_and_passes_message(tmp_path):
    hooks = tmp_path / "hooks"
    write_hook(hooks, "pre_tool", "echo 'todo bien'\nexit 0\n")
    result = HookRunner(dir=hooks).run_pre_tool("bash", {"command": "ls"})
    assert result.blocked is False
    assert result.message == "todo bien"


def test_nonzero_exit_blocks_with_stderr(tmp_path):
    hooks = tmp_path / "hooks"
    write_hook(hooks, "pre_tool", "echo 'no toques prod' >&2\nexit 1\n")
    result = HookRunner(dir=hooks).run_pre_tool("write_file", {"path": "prod.env"})
    assert result.blocked is True
    assert result.message == "no toques prod"


def test_nonzero_exit_without_output_has_fallback_message(tmp_path):
    hooks = tmp_path / "hooks"
    write_hook(hooks, "pre_tool", "exit 3\n")
    result = HookRunner(dir=hooks).run_pre_tool("bash", {})
    assert result.blocked is True
    assert "exit 3" in result.message


def test_hook_receives_payload_on_stdin(tmp_path):
    hooks = tmp_path / "hooks"
    out = tmp_path / "captured.json"
    write_hook(hooks, "pre_tool", f"cat > {out}\nexit 0\n")
    HookRunner(dir=hooks).run_pre_tool("bash", {"command": "ls -la"})
    payload = json.loads(out.read_text())
    assert payload["hook"] == "pre_tool"
    assert payload["tool"] == "bash"
    assert payload["arguments"] == {"command": "ls -la"}
    assert payload["cwd"]


def test_timeout_blocks(tmp_path):
    hooks = tmp_path / "hooks"
    write_hook(hooks, "pre_tool", "sleep 5\n")
    runner = HookRunner(dir=hooks)
    runner.timeout = 1
    result = runner.run_pre_tool("bash", {})
    assert result.blocked is True
    assert "timed out" in result.message


def test_non_executable_hook_blocks(tmp_path):
    hooks = tmp_path / "hooks"
    write_hook(hooks, "pre_tool", "exit 0\n", executable=False)
    result = HookRunner(dir=hooks).run_pre_tool("bash", {})
    assert result.blocked is True


def test_post_tool_returns_output(tmp_path):
    hooks = tmp_path / "hooks"
    write_hook(hooks, "post_tool", "echo 'anotado'\nexit 0\n")
    assert HookRunner(dir=hooks).run_post_tool("bash", {}, "salida") == "anotado"


def test_post_tool_never_blocks_on_failure(tmp_path):
    hooks = tmp_path / "hooks"
    write_hook(hooks, "post_tool", "echo boom >&2\nexit 9\n")
    assert HookRunner(dir=hooks).run_post_tool("bash", {}, "x") == ""


def test_post_tool_timeout_returns_empty(tmp_path):
    hooks = tmp_path / "hooks"
    write_hook(hooks, "post_tool", "sleep 5\n")
    runner = HookRunner(dir=hooks)
    runner.timeout = 1
    assert runner.run_post_tool("bash", {}, "x") == ""


def test_post_tool_receives_result(tmp_path):
    hooks = tmp_path / "hooks"
    out = tmp_path / "captured.json"
    write_hook(hooks, "post_tool", f"cat > {out}\nexit 0\n")
    HookRunner(dir=hooks).run_post_tool("read_file", {"path": "a"}, "contenido")
    payload = json.loads(out.read_text())
    assert payload["result"] == "contenido"
    assert payload["hook"] == "post_tool"


def test_pre_tool_blocks_on_unexpected_launch_error(tmp_path, monkeypatch):
    """Any launch failure that is neither timeout nor a missing hook must still
    fail closed — a guard that cannot run is a closed gate."""
    hooks = tmp_path / "hooks"
    write_hook(hooks, "pre_tool", "exit 0\n")
    runner = HookRunner(dir=hooks)

    def boom(*a, **k):
        raise MemoryError("out of memory")

    monkeypatch.setattr("local_code.hooks.subprocess.run", boom)
    result = runner.run_pre_tool("bash", {})
    assert result.blocked is True
    assert "failed" in result.message


def test_post_tool_swallows_unexpected_launch_error(tmp_path, monkeypatch):
    hooks = tmp_path / "hooks"
    write_hook(hooks, "post_tool", "exit 0\n")
    runner = HookRunner(dir=hooks)

    def boom(*a, **k):
        raise MemoryError("out of memory")

    monkeypatch.setattr("local_code.hooks.subprocess.run", boom)
    assert runner.run_post_tool("bash", {}, "x") == ""
