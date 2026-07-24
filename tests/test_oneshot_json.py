from __future__ import annotations

import argparse
import io
import json

from rich.console import Console

from local_code.cli import AppContext, build_oneshot_text, run_oneshot_json
from local_code.config import Config
from local_code.session import Session


class FakeDetector:
    def supports_tools(self, model):
        return True


class FakeClient:
    name = "fake"

    def show(self, model):
        return {}

    def chat(self, *a, **k):
        # No tokens, no tool calls -> the turn produces an empty response.
        return iter([])


class FakeStore:
    def __init__(self):
        self.saved = []

    def save(self, *a):
        self.saved.append(a)


class FakeMCP:
    def __init__(self):
        self.shut = False

    def shutdown(self):
        self.shut = True


def make_ctx(**overrides):
    args = argparse.Namespace(yolo=True, json=True)
    ctx = AppContext(
        args=args,
        cfg=Config(),
        console=Console(file=io.StringIO(), force_terminal=False),
        client=FakeClient(),
        detector=FakeDetector(),
        model="m",
        plan_mode=False,
        mcp_manager=FakeMCP(),
        store=FakeStore(),
        checkpoints=None,
        session=Session(system_prompt="x"),
        session_id="sid-1",
        spawn_factory=None,
    )
    for k, v in overrides.items():
        setattr(ctx, k, v)
    return ctx


def test_build_oneshot_text_combinations():
    assert build_oneshot_text([], None) is None
    assert build_oneshot_text(["fix", "bug"], None) == "fix bug"
    assert build_oneshot_text([], "piped") == "piped"
    assert build_oneshot_text(["explain"], "trace") == "explain\n\ntrace"


def test_run_oneshot_json_emits_json_on_stdout(capsys):
    ctx = make_ctx()
    code = run_oneshot_json(ctx, "hello")
    assert code == 0
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["ok"] is True
    assert payload["model"] == "m"
    assert payload["session_id"] == "sid-1"
    assert "response" in payload
    assert ctx.mcp_manager.shut is True
    assert ctx.store.saved  # session was saved


def test_run_oneshot_json_reports_backend_error(capsys):
    from local_code.backends import ModelNotFoundError
    from local_code.cli import EXIT_MODEL_NOT_FOUND

    class BoomClient(FakeClient):
        def chat(self, *a, **k):
            raise ModelNotFoundError("no such model")

    ctx = make_ctx(client=BoomClient())
    code = run_oneshot_json(ctx, "hello")
    assert code == EXIT_MODEL_NOT_FOUND
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["ok"] is False
    assert payload["error_type"] == "ModelNotFoundError"
