from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from local_code import react, tools
from local_code.tools import ToolContext

DEFAULT_SYSTEM_PROMPT = (
    "You are local-code, a coding agent running in the user's terminal.\n"
    "Working directory: {cwd}\n"
    "Use the available tools to inspect and modify files and to run commands.\n"
    "Read the relevant files before editing them. Keep answers short.\n"
    "For multi-step tasks, call set_todos first to plan, and update item "
    "statuses as you progress."
)

PLAN_MODE_INSTRUCTION = (
    "\n\n# Plan mode\n"
    "You are currently in plan mode. Use read-only tools (read_file, list_dir, "
    "glob, grep, set_todos) to investigate the codebase. "
    "Do NOT attempt edits, writes, bash commands, or any other side-effecting tool. "
    "When you have enough information, output a clear, numbered, step-by-step plan "
    "describing every change you would make. End your response with the plan — "
    "the user will review it and type /approve to execute it."
)

DECLINED = "User declined to run this tool."


@dataclass
class AgentConfig:
    model: str
    max_iterations: int = 25
    bash_timeout: int = 120
    yolo: bool = False
    plan_mode: bool = False


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
        on_stream_start: Callable[[], None] | None = None,
        on_stream_end: Callable[[], None] | None = None,
        compactor=None,
        permission_store=None,
        on_todos=None,
        hook_runner=None,
        checkpoint_store=None,
    ):
        self.client = client
        self.session = session
        self.config = config
        self.use_native = use_native
        self.confirm = confirm or (lambda name, preview: True)
        self.on_token = on_token or (lambda token: None)
        self.notify = notify or (lambda message: None)
        self.on_stream_start = on_stream_start or (lambda: None)
        self.on_stream_end = on_stream_end or (lambda: None)
        self.compactor = compactor
        self.permission_store = permission_store
        self.hook_runner = hook_runner
        self.checkpoint_store = checkpoint_store
        self.context = ToolContext(
            bash_timeout=config.bash_timeout, on_todos=on_todos
        )

    def _effective_system_prompt(self) -> str:
        base = self.session.system_prompt or ""
        if self.config.plan_mode:
            return base + PLAN_MODE_INSTRUCTION
        return base

    def run_turn(self, user_input: str) -> str:
        if self.compactor is not None:
            self.compactor.maybe_compact(self.session)
        self.session.add({"role": "user", "content": user_input})
        if self.use_native:
            return self._run_native()
        return self._run_react()

    def _stream(self, messages: list[dict], tools_param: list[dict] | None):
        content = ""
        tool_calls: list[dict] = []
        self.on_stream_start()
        try:
            for chunk in self.client.chat(self.config.model, messages, tools=tools_param):
                msg = chunk.get("message", {})
                token = msg.get("content", "")
                if token:
                    content += token
                    self.on_token(token)
                tool_calls.extend(msg.get("tool_calls") or [])
        finally:
            self.on_stream_end()
        return content, tool_calls

    def _execute(self, name: str, arguments: dict) -> str:
        # Plan-mode gate: block side-effecting tools regardless of yolo.
        if self.config.plan_mode and tools.requires_confirmation(name):
            return (
                f"Plan mode: not running '{name}'. "
                f"Describe this change in your plan instead of executing it."
            )

        if not self.config.yolo and tools.requires_confirmation(name):
            pre_allowed = False
            if self.permission_store is not None:
                try:
                    pre_allowed = self.permission_store.is_allowed(name, arguments)
                except Exception:
                    pre_allowed = False
            if not pre_allowed:
                try:
                    preview = tools.get_preview(name, arguments)
                except Exception as e:
                    preview = f"{name} (malformed call: {type(e).__name__}: {e})"
                decision = self.confirm(name, preview)
                if decision is True:
                    decision = "yes"
                elif decision is False:
                    decision = "no"
                if decision not in ("yes", "always"):
                    return DECLINED
                if decision == "always" and self.permission_store is not None:
                    try:
                        self.permission_store.allow(name, arguments)
                    except Exception:
                        pass
        if self.hook_runner is not None:
            hook = self.hook_runner.run_pre_tool(name, arguments)
            if hook.blocked:
                return f"Blocked by hook: {hook.message}"
            if hook.message:
                self.notify(hook.message)
        if self.checkpoint_store is not None:
            self.checkpoint_store.snapshot(name, arguments)
        self.notify(f"→ {name}({json.dumps(arguments, ensure_ascii=False)[:120]})")
        result = tools.execute(name, arguments, self.context)
        if self.hook_runner is not None:
            note = self.hook_runner.run_post_tool(name, arguments, result)
            if note:
                self.notify(note)
        return result

    def _run_native(self) -> str:
        schemas = tools.tool_schemas()
        content = ""
        for _ in range(self.config.max_iterations):
            # Build messages with plan-mode system prompt override when needed.
            messages = self._native_messages()
            content, tool_calls = self._stream(messages, schemas)
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

    def _native_messages(self) -> list[dict]:
        """Build the message list for a native tool-calling turn.

        In plan mode the system prompt is augmented, so we rebuild it rather
        than relying on session.messages (which uses the unaugmented prompt).
        """
        prompt = self._effective_system_prompt()
        if prompt:
            return [{"role": "system", "content": prompt}] + self.session.history
        return list(self.session.history)

    def _react_messages(self) -> list[dict]:
        system = react.build_system_prompt(
            self._effective_system_prompt(), tools.tool_schemas()
        )
        return [{"role": "system", "content": system}] + self.session.history

    def _run_react(self) -> str:
        failures = 0
        content = ""
        for _ in range(self.config.max_iterations):
            content, _ = self._stream(self._react_messages(), None)
            self.session.add({"role": "assistant", "content": content})
            try:
                calls = react.parse_tool_calls(content)
            except react.ToolCallParseError as e:
                failures += 1
                if failures > 2:
                    self.notify("Aborting: 3 consecutive invalid tool calls.")
                    return content
                self.session.add(
                    {"role": "user", "content": react.format_observation(f"Error: {e}")}
                )
                continue
            unknown = [c for c in calls if tools.get_tool(c.name) is None]
            if unknown:
                failures += 1
                if failures > 2:
                    self.notify("Aborting: 3 consecutive invalid tool calls.")
                    return content
                obs = "\n\n".join(
                    react.format_observation(f"Error: unknown tool '{c.name}'")
                    for c in unknown
                )
                self.session.add({"role": "user", "content": obs})
                continue
            if not calls:
                return content
            failures = 0
            observations = [
                react.format_observation(self._execute(c.name, c.arguments))
                for c in calls
            ]
            self.session.add({"role": "user", "content": "\n\n".join(observations)})
        self.notify(f"Reached max iterations ({self.config.max_iterations}); stopping.")
        return content
