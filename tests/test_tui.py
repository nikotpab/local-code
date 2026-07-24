from __future__ import annotations

import asyncio
import pytest

from rich.console import Console

from local_code.agent import Agent, AgentConfig
from local_code.capabilities import CapabilityDetector
from local_code.checkpoints import CheckpointStore
from local_code.cli import AppContext, parse_args
from local_code.config import Config
from local_code.mcp import MCPManager
from local_code.session import Session
from local_code.session_store import SessionStore
from local_code.tui.app import LocalCodeApp
from local_code.tui.bridge import ThreadSafeConfirmationBridge
from local_code.tui.widgets import (
    ActivityPane,
    ConfirmationModal,
    ConversationPane,
    HeaderBar,
)


class FakeClient:
    def __init__(self, responses: list[dict] | None = None):
        self.name = "fake-backend"
        self.responses = responses or [{"message": {"content": "Hello world from fake-backend!"}}]

    def chat(self, model: str, messages: list[dict], tools: list[dict] | None = None):
        for r in self.responses:
            yield r


class FakeDetector:
    def supports_tools(self, model: str) -> bool:
        return True


@pytest.fixture
def dummy_app_ctx(tmp_path) -> AppContext:
    args = parse_args([])
    cfg = Config()
    console = Console(quiet=True)
    client = FakeClient()
    detector = FakeDetector()
    mcp_manager = MCPManager()
    store = SessionStore(dir=tmp_path / "sessions")
    checkpoints = CheckpointStore(dir=tmp_path / "checkpoints")
    session = Session(system_prompt="system test")
    session_id = store.new_id()

    return AppContext(
        args=args,
        cfg=cfg,
        console=console,
        client=client,
        detector=detector,
        model="fake-model",
        plan_mode=False,
        mcp_manager=mcp_manager,
        store=store,
        checkpoints=checkpoints,
        session=session,
        session_id=session_id,
        spawn_factory=lambda task, model: "Subagent report",
    )


def test_tui_app_mount_and_widgets(dummy_app_ctx):
    async def _test():
        app = LocalCodeApp(dummy_app_ctx)
        async with app.run_test() as pilot:
            assert app.query_one(HeaderBar) is not None
            assert app.query_one(ConversationPane) is not None
            assert app.query_one(ActivityPane) is not None

            header = app.query_one(HeaderBar)
            assert header.model == "fake-model"

    asyncio.run(_test())


def test_tui_slash_commands(dummy_app_ctx):
    async def _test():
        app = LocalCodeApp(dummy_app_ctx)
        async with app.run_test() as pilot:
            # Test /plan command
            app.execute_command("plan", None, "/plan")
            assert app.agent.config.plan_mode is True

            # Test /help command
            app.execute_command("help", None, "/help")
            await pilot.pause()

            # Test /clear command
            app.execute_command("clear", None, "/clear")
            assert len(app.app_ctx.session.history) == 0

    asyncio.run(_test())


def test_tui_toggle_activity_pane(dummy_app_ctx):
    async def _test():
        app = LocalCodeApp(dummy_app_ctx)
        async with app.run_test() as pilot:
            activity = app.query_one(ActivityPane)
            assert not activity.has_class("hidden")

            app.action_toggle_activity()
            assert activity.has_class("hidden")

            app.action_toggle_activity()
            assert not activity.has_class("hidden")

    asyncio.run(_test())


def test_tui_user_submit_and_turn_dispatch(dummy_app_ctx):
    async def _test():
        app = LocalCodeApp(dummy_app_ctx)
        async with app.run_test() as pilot:
            app.on_user_submit("hi agent")
            await pilot.pause()

            # Check session has user message
            assert any(m.get("content") == "hi agent" for m in app.app_ctx.session.history)

    asyncio.run(_test())


def test_tui_confirmation_bridge_yes(dummy_app_ctx):
    called = []

    def mock_show_modal(name, preview, callback):
        called.append((name, preview))
        callback("yes")

    class FakeApp:
        def call_from_thread(self, fn, *args):
            fn(*args)

    bridge = ThreadSafeConfirmationBridge(FakeApp(), mock_show_modal)
    result = bridge.confirm("edit_file", "--- a/x\n+++ b/x\n")
    assert result == "yes"
    assert len(called) == 1
    assert called[0][0] == "edit_file"


def test_tui_confirmation_bridge_fail_closed(dummy_app_ctx):
    def mock_show_modal(name, preview, callback):
        callback("invalid_choice")

    class FakeApp:
        def call_from_thread(self, fn, *args):
            fn(*args)

    bridge = ThreadSafeConfirmationBridge(FakeApp(), mock_show_modal)
    result = bridge.confirm("bash", "rm -rf /")
    assert result == "no"


def test_tui_renders_tool_activity(dummy_app_ctx):
    async def _test():
        app = LocalCodeApp(dummy_app_ctx)
        async with app.run_test() as pilot:
            conv = app.query_one(ConversationPane)
            before = len(list(conv.children))
            app._on_tool_start("read_file", {"path": "a.py"})
            app._on_tool_end("read_file", "42 chars")
            await pilot.pause()
            assert len(list(conv.children)) > before

    asyncio.run(_test())
