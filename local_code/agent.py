from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from local_code import react, tools
from local_code.tools import ToolContext

DEFAULT_SYSTEM_PROMPT = (
    "You are local-code, a coding agent running in the user's terminal.\n"
    "Working directory: {cwd}\n"
    "Use the available tools to inspect and modify files and to run commands.\n"
    "Read the relevant files before editing them. Keep answers short."
)
DECLINED = "User declined to run this tool."


@dataclass
class AgentConfig:
    model: str
    max_iterations: int = 25
    bash_timeout: int = 120
    yolo: bool = False


class Agent:
    def __init__(
        self,
        client,
        session,
        config: AgentConfig,
        use_native: bool,
        confirm: Callable[[str, str], bool] | None = None,
        on_token: Callable[[str], None] | None = None,
        notify: Callable[[str], None] | None = None,
    ):
        self.client = client
        self.session = session
        self.config = config
        self.use_native = use_native
        self.confirm = confirm or (lambda name, preview: True)
        self.on_token = on_token or (lambda token: None)
        self.notify = notify or (lambda message: None)
        self.context = ToolContext(bash_timeout=config.bash_timeout)

    def run_turn(self, user_input: str) -> str:
        self.session.add({"role": "user", "content": user_input})
        if self.use_native:
            return self._run_native()
        return self._run_react()

    def _stream(self, messages: list[dict], tools_param: list[dict] | None):
        content = ""
        tool_calls: list[dict] = []
        for chunk in self.client.chat(self.config.model, messages, tools=tools_param):
            msg = chunk.get("message", {})
            token = msg.get("content", "")
            if token:
                content += token
                self.on_token(token)
            tool_calls.extend(msg.get("tool_calls") or [])
        return content, tool_calls

    def _execute(self, name: str, arguments: dict) -> str:
        if not self.config.yolo and tools.requires_confirmation(name):
            preview = tools.get_preview(name, arguments)
            if not self.confirm(name, preview):
                return DECLINED
        self.notify(f"→ {name}({json.dumps(arguments, ensure_ascii=False)[:120]})")
        return tools.execute(name, arguments, self.context)

    def _run_native(self) -> str:
        schemas = tools.tool_schemas()
        content = ""
        for _ in range(self.config.max_iterations):
            content, tool_calls = self._stream(self.session.messages, schemas)
            if not tool_calls:
                self.session.add({"role": "assistant", "content": content})
                return content
            self.session.add(
                {"role": "assistant", "content": content, "tool_calls": tool_calls}
            )
            for call in tool_calls:
                fn = call.get("function", {})
                name = fn.get("name", "")
                arguments = fn.get("arguments") or {}
                result = self._execute(name, arguments)
                self.session.add(
                    {"role": "tool", "tool_name": name, "content": result}
                )
        self.notify(f"Reached max iterations ({self.config.max_iterations}); stopping.")
        return content

    def _run_react(self) -> str:
        raise NotImplementedError  # Task 13
