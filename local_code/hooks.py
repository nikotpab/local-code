from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

HOOKS_DIR = Path.home() / ".local-code" / "hooks"
HOOK_TIMEOUT = 10
MAX_RESULT_CHARS = 10_000


@dataclass
class HookResult:
    blocked: bool
    message: str


class HookRunner:
    def __init__(self, dir: Path | None = None):
        self.dir = dir or HOOKS_DIR
        self.timeout = HOOK_TIMEOUT

    def _run(self, name: str, payload: dict):
        path = self.dir / name
        if not path.is_file():
            return None
        return subprocess.run(
            [str(path)],
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )

    def run_pre_tool(self, tool_name: str, arguments: dict) -> HookResult:
        payload = {
            "hook": "pre_tool",
            "tool": tool_name,
            "arguments": arguments,
            "cwd": str(Path.cwd()),
        }
        try:
            proc = self._run("pre_tool", payload)
        except subprocess.TimeoutExpired:
            return HookResult(True, "pre_tool hook timed out")
        except OSError as e:
            return HookResult(True, f"pre_tool hook failed: {e}")
        if proc is None:
            return HookResult(False, "")
        if proc.returncode == 0:
            return HookResult(False, proc.stdout.strip())
        message = (proc.stderr or proc.stdout).strip()
        return HookResult(
            True, message or f"blocked by pre_tool hook (exit {proc.returncode})"
        )

    def run_post_tool(self, tool_name: str, arguments: dict, result: str) -> str:
        payload = {
            "hook": "post_tool",
            "tool": tool_name,
            "arguments": arguments,
            "result": result[:MAX_RESULT_CHARS],
            "cwd": str(Path.cwd()),
        }
        try:
            proc = self._run("post_tool", payload)
        except (subprocess.TimeoutExpired, OSError):
            return ""
        if proc is None or proc.returncode != 0:
            return ""
        return proc.stdout.strip()
