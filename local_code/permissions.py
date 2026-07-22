from __future__ import annotations

from pathlib import Path

import yaml

PERMISSIONS_PATH = Path.home() / ".local-code" / "permissions.yaml"


class PermissionStore:
    def __init__(self, path: Path | None = None):
        self.path = path or PERMISSIONS_PATH
        self.allowed_tools: list[str] = []
        self.allowed_bash_prefixes: list[str] = []
        try:
            data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            data = {}
        if isinstance(data, dict):
            tools = data.get("allowed_tools", [])
            prefixes = data.get("allowed_bash_prefixes", [])
            if isinstance(tools, list):
                self.allowed_tools = [t for t in tools if isinstance(t, str)]
            if isinstance(prefixes, list):
                self.allowed_bash_prefixes = [
                    p for p in prefixes if isinstance(p, str) and p
                ]

    def is_allowed(self, tool_name: str, arguments: dict) -> bool:
        if tool_name == "bash":
            command = arguments.get("command", "")
            return any(command.startswith(p) for p in self.allowed_bash_prefixes)
        return tool_name in self.allowed_tools

    def allow(self, tool_name: str, arguments: dict) -> None:
        if tool_name == "bash":
            command = arguments.get("command", "")
            if command and command not in self.allowed_bash_prefixes:
                self.allowed_bash_prefixes.append(command)
        elif tool_name not in self.allowed_tools:
            self.allowed_tools.append(tool_name)
        self._save()

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                yaml.safe_dump(
                    {
                        "allowed_tools": self.allowed_tools,
                        "allowed_bash_prefixes": self.allowed_bash_prefixes,
                    },
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass
