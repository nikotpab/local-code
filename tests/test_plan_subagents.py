from __future__ import annotations

"""Tests for Plan Mode and Subagents features."""


from local_code.agent import PLAN_MODE_INSTRUCTION, Agent, AgentConfig
from local_code.cli import (
    _last_assistant_message,
    handle_command,
    make_spawn_factory,
    parse_args,
)
from local_code.session import Session
from local_code.tools import spawn_agent as spawn_agent_mod
from local_code.tools.context import ToolContext
from tests.conftest import FakeClient, text_chunks, tool_call_chunks

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_agent(client, *, plan_mode=False, yolo=False, system="base", **kwargs):
    cfg = AgentConfig(model="m", yolo=yolo, plan_mode=plan_mode)
    return Agent(client, Session(system_prompt=system), cfg, use_native=True, **kwargs)


# ---------------------------------------------------------------------------
# Plan mode: gate in _execute
# ---------------------------------------------------------------------------

class TestPlanModeGate:
    def test_mutating_tool_blocked_in_plan_mode(self, tmp_path, monkeypatch):
        """write_file (requires_confirmation=True) must be blocked in plan mode."""
        monkeypatch.chdir(tmp_path)
        client = FakeClient([
            tool_call_chunks("write_file", {"path": "x.txt", "content": "hi"}),
            text_chunks("plan written"),
        ])
        agent = make_agent(client, plan_mode=True)
        agent.run_turn("investigate")
        # File must NOT have been created.
        assert not (tmp_path / "x.txt").exists()
        # The tool result in history must contain the plan-mode message.
        tool_results = [m["content"] for m in agent.session.history if m.get("role") == "tool"]
        assert any("Plan mode" in r for r in tool_results)

    def test_mutating_tool_blocked_message_contains_tool_name(self):
        """Block message must name the tool that was blocked."""
        agent = make_agent(FakeClient([]), plan_mode=True)
        result = agent._execute("bash", {"command": "rm -rf /"})
        assert "bash" in result
        assert "Plan mode" in result

    def test_read_only_tool_runs_in_plan_mode(self, tmp_path, monkeypatch):
        """read_file (requires_confirmation=False) must run normally in plan mode."""
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "notes.txt"
        target.write_text("hello plan")
        client = FakeClient([
            tool_call_chunks("read_file", {"path": str(target)}),
            text_chunks("I read it"),
        ])
        agent = make_agent(client, plan_mode=True)
        agent.run_turn("read the file")
        tool_results = [m["content"] for m in agent.session.history if m.get("role") == "tool"]
        assert any("hello plan" in r for r in tool_results)

    def test_plan_mode_overrides_yolo(self, tmp_path, monkeypatch):
        """Even with yolo=True, plan mode must block mutating tools."""
        monkeypatch.chdir(tmp_path)
        client = FakeClient([
            tool_call_chunks("write_file", {"path": "y.txt", "content": "data"}),
            text_chunks("done"),
        ])
        agent = make_agent(client, plan_mode=True, yolo=True)
        agent.run_turn("do it")
        assert not (tmp_path / "y.txt").exists()

    def test_plan_mode_off_allows_mutating_tool(self, tmp_path, monkeypatch):
        """With plan_mode=False and yolo=True, mutating tools run freely."""
        monkeypatch.chdir(tmp_path)
        client = FakeClient([
            tool_call_chunks("write_file", {"path": "z.txt", "content": "ok"}),
            text_chunks("done"),
        ])
        agent = make_agent(client, plan_mode=False, yolo=True)
        agent.run_turn("write it")
        assert (tmp_path / "z.txt").read_text() == "ok"


# ---------------------------------------------------------------------------
# Plan mode: system prompt
# ---------------------------------------------------------------------------

class TestPlanModeSystemPrompt:
    def test_system_prompt_appended_in_plan_mode(self):
        agent = make_agent(FakeClient([]), plan_mode=True, system="base prompt")
        effective = agent._effective_system_prompt()
        assert effective.startswith("base prompt")
        assert PLAN_MODE_INSTRUCTION in effective

    def test_system_prompt_unchanged_when_not_plan_mode(self):
        agent = make_agent(FakeClient([]), plan_mode=False, system="base prompt")
        effective = agent._effective_system_prompt()
        assert effective == "base prompt"
        assert PLAN_MODE_INSTRUCTION not in effective

    def test_plan_instruction_reaches_model_messages(self):
        """_native_messages() must carry the plan instruction in plan mode."""
        agent = make_agent(FakeClient([]), plan_mode=True, system="sys")
        msgs = agent._native_messages()
        system_msg = next((m for m in msgs if m.get("role") == "system"), None)
        assert system_msg is not None
        assert PLAN_MODE_INSTRUCTION in system_msg["content"]

    def test_react_messages_carry_plan_instruction(self):
        """_react_messages() must also inject the plan instruction."""
        agent = make_agent(FakeClient([]), plan_mode=True, system="sys")
        msgs = agent._react_messages()
        combined = " ".join(m.get("content", "") for m in msgs if m.get("role") == "system")
        assert "plan mode" in combined.lower() or PLAN_MODE_INSTRUCTION in combined


# ---------------------------------------------------------------------------
# CLI: handle_command /plan and /approve
# ---------------------------------------------------------------------------

class TestHandleCommandPlanApprove:
    def test_plan_command(self):
        assert handle_command("/plan") == ("plan", None)

    def test_approve_command(self):
        assert handle_command("/approve") == ("approve", None)

    def test_plan_is_not_chat(self):
        action, _ = handle_command("/plan")
        assert action != "chat"

    def test_approve_is_not_custom(self):
        action, _ = handle_command("/approve")
        assert action != "custom"


# ---------------------------------------------------------------------------
# CLI: --plan parse
# ---------------------------------------------------------------------------

class TestParsePlan:
    def test_plan_flag_default_false(self):
        args = parse_args([])
        assert args.plan is False

    def test_plan_flag_set(self):
        args = parse_args(["--plan"])
        assert args.plan is True

    def test_plan_with_prompt(self):
        args = parse_args(["--plan", "investigate", "the", "codebase"])
        assert args.plan is True
        assert args.prompt == ["investigate", "the", "codebase"]


# ---------------------------------------------------------------------------
# CLI: _last_assistant_message helper
# ---------------------------------------------------------------------------

class TestLastAssistantMessage:
    def test_returns_last_assistant_content(self):
        session = Session()
        session.add({"role": "user", "content": "hello"})
        session.add({"role": "assistant", "content": "here is the plan"})
        assert _last_assistant_message(session) == "here is the plan"

    def test_returns_most_recent(self):
        session = Session()
        session.add({"role": "assistant", "content": "first"})
        session.add({"role": "assistant", "content": "second"})
        assert _last_assistant_message(session) == "second"

    def test_returns_none_when_no_assistant_messages(self):
        session = Session()
        session.add({"role": "user", "content": "hello"})
        assert _last_assistant_message(session) is None

    def test_returns_none_when_empty_history(self):
        assert _last_assistant_message(Session()) is None

    def test_skips_empty_content(self):
        session = Session()
        session.add({"role": "assistant", "content": ""})
        session.add({"role": "assistant", "content": "the plan"})
        assert _last_assistant_message(session) == "the plan"


# ---------------------------------------------------------------------------
# spawn_agent tool: run()
# ---------------------------------------------------------------------------

class TestSpawnAgentRun:
    def test_returns_error_when_spawn_is_none(self):
        ctx = ToolContext()
        assert ctx.spawn is None
        result = spawn_agent_mod.run({"task": "do something"}, ctx)
        assert result.startswith("Error:")
        assert "not available" in result

    def test_calls_spawn_with_task_and_model(self):
        calls = []

        def fake_spawn(task, model):
            calls.append((task, model))
            return "report text"

        ctx = ToolContext(spawn=fake_spawn)
        result = spawn_agent_mod.run({"task": "investigate x", "model": "llama3"}, ctx)
        assert result == "report text"
        assert calls == [("investigate x", "llama3")]

    def test_model_defaults_to_none_when_omitted(self):
        calls = []

        def fake_spawn(task, model):
            calls.append((task, model))
            return "ok"

        ctx = ToolContext(spawn=fake_spawn)
        spawn_agent_mod.run({"task": "investigate x"}, ctx)
        assert calls[0][1] is None

    def test_empty_task_returns_error(self):
        ctx = ToolContext(spawn=lambda t, m: "ok")
        result = spawn_agent_mod.run({"task": ""}, ctx)
        assert result.startswith("Error:")

    def test_requires_confirmation_is_false(self):
        assert spawn_agent_mod.REQUIRES_CONFIRMATION is False

    def test_name_is_spawn_agent(self):
        assert spawn_agent_mod.NAME == "spawn_agent"

    def test_parameters_has_required_task(self):
        assert "task" in spawn_agent_mod.PARAMETERS["required"]

    def test_parameters_model_is_optional(self):
        assert "model" in spawn_agent_mod.PARAMETERS["properties"]
        assert "model" not in spawn_agent_mod.PARAMETERS.get("required", [])


# ---------------------------------------------------------------------------
# Subagent factory: make_spawn_factory
# ---------------------------------------------------------------------------

class TestSpawnFactory:
    def _make_console(self):
        import io

        from rich.console import Console
        return Console(file=io.StringIO(), highlight=False)

    def test_factory_returns_string(self, monkeypatch):
        """make_spawn_factory should produce a callable that returns a string."""
        console = self._make_console()
        client = FakeClient([text_chunks("subagent report")])
        cfg_mock = type("Cfg", (), {"bash_timeout_seconds": 120, "max_iterations": 5,
                                    "context_window": None})()
        # Patch CapabilityDetector to avoid network calls.
        monkeypatch.setattr(
            "local_code.cli.CapabilityDetector",
            lambda c: type("D", (), {"supports_tools": lambda self, m: True})(),
        )
        factory = make_spawn_factory(client, cfg_mock, "base-model", console)
        result = factory("investigate this", None)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_factory_result_prefixed_with_subagent(self, monkeypatch):
        console = self._make_console()
        client = FakeClient([text_chunks("my findings")])
        cfg_mock = type("Cfg", (), {"bash_timeout_seconds": 120, "max_iterations": 5,
                                    "context_window": None})()
        monkeypatch.setattr(
            "local_code.cli.CapabilityDetector",
            lambda c: type("D", (), {"supports_tools": lambda self, m: True})(),
        )
        factory = make_spawn_factory(client, cfg_mock, "base-model", console)
        result = factory("investigate", None)
        # The subagent report is prefixed so the parent can identify it.
        assert "Subagent" in result or "subagent" in result or "my findings" in result

    def test_subagent_context_spawn_is_none(self, monkeypatch):
        """The subagent's ToolContext.spawn must be None (recursion guard)."""
        console = self._make_console()
        captured_spawn = []

        class CapturingFakeClient:
            def chat(self, model, messages, tools=None):
                return iter([{"message": {"role": "assistant", "content": "done"}, "done": True}])

        cfg_mock = type("Cfg", (), {"bash_timeout_seconds": 120, "max_iterations": 1,
                                    "context_window": None})()
        monkeypatch.setattr(
            "local_code.cli.CapabilityDetector",
            lambda c: type("D", (), {"supports_tools": lambda self, m: True})(),
        )

        original_agent_init = Agent.__init__

        def capturing_init(self, *args, **kwargs):
            original_agent_init(self, *args, **kwargs)
            captured_spawn.append(self.context.spawn)

        monkeypatch.setattr(Agent, "__init__", capturing_init)
        factory = make_spawn_factory(CapturingFakeClient(), cfg_mock, "m", console)
        factory("task", None)
        # The subagent's context.spawn must be explicitly None.
        assert None in captured_spawn

    def test_subagent_is_read_only(self, monkeypatch, tmp_path):
        """Subagent must not be able to write files even in the factory."""
        console = self._make_console()
        # Attempt a write_file call from inside the subagent.
        client = FakeClient([
            tool_call_chunks("write_file", {"path": str(tmp_path / "evil.txt"), "content": "no"}),
            text_chunks("blocked"),
        ])
        cfg_mock = type("Cfg", (), {"bash_timeout_seconds": 120, "max_iterations": 5,
                                    "context_window": None})()
        monkeypatch.setattr(
            "local_code.cli.CapabilityDetector",
            lambda c: type("D", (), {"supports_tools": lambda self, m: True})(),
        )
        factory = make_spawn_factory(client, cfg_mock, "m", console)
        factory("try to write", None)
        # File must NOT exist.
        assert not (tmp_path / "evil.txt").exists()

    def test_subagent_failure_returns_error_string(self, monkeypatch):
        """A subagent that throws must return an 'Error: ...' string."""
        console = self._make_console()

        class BrokenClient:
            def chat(self, *a, **k):
                raise RuntimeError("network gone")

        cfg_mock = type("Cfg", (), {"bash_timeout_seconds": 120, "max_iterations": 1,
                                    "context_window": None})()
        monkeypatch.setattr(
            "local_code.cli.CapabilityDetector",
            lambda c: type("D", (), {"supports_tools": lambda self, m: True})(),
        )
        factory = make_spawn_factory(BrokenClient(), cfg_mock, "m", console)
        result = factory("task", None)
        assert result.startswith("Error:")
        assert "network gone" in result

    def test_subagent_uses_provided_model(self, monkeypatch):
        """When a model is specified, the subagent AgentConfig uses it."""
        console = self._make_console()
        used_models = []

        class ModelCapturingClient:
            def chat(self, model, messages, tools=None):
                used_models.append(model)
                return iter([{"message": {"role": "assistant", "content": "ok"}, "done": True}])

        cfg_mock = type("Cfg", (), {"bash_timeout_seconds": 120, "max_iterations": 1,
                                    "context_window": None})()
        monkeypatch.setattr(
            "local_code.cli.CapabilityDetector",
            lambda c: type("D", (), {"supports_tools": lambda self, m: True})(),
        )
        factory = make_spawn_factory(ModelCapturingClient(), cfg_mock, "parent-model", console)
        factory("task", "override-model")
        assert "override-model" in used_models

    def test_subagent_uses_parent_model_when_none(self, monkeypatch):
        """When model is None, the subagent uses the parent model."""
        console = self._make_console()
        used_models = []

        class ModelCapturingClient:
            def chat(self, model, messages, tools=None):
                used_models.append(model)
                return iter([{"message": {"role": "assistant", "content": "ok"}, "done": True}])

        cfg_mock = type("Cfg", (), {"bash_timeout_seconds": 120, "max_iterations": 1,
                                    "context_window": None})()
        monkeypatch.setattr(
            "local_code.cli.CapabilityDetector",
            lambda c: type("D", (), {"supports_tools": lambda self, m: True})(),
        )
        factory = make_spawn_factory(ModelCapturingClient(), cfg_mock, "parent-model", console)
        factory("task", None)
        assert "parent-model" in used_models
