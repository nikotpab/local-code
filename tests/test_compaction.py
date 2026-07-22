from __future__ import annotations

from local_code.compaction import (
    DEFAULT_CONTEXT_WINDOW,
    KEEP_RECENT,
    Compactor,
    detect_context_window,
    estimate_tokens,
)
from local_code.session import Session
from tests.conftest import FakeClient, text_chunks


class FakeShowClient:
    def __init__(self, info=None, raise_exc=False):
        self.info = info or {}
        self.raise_exc = raise_exc

    def show(self, model):
        if self.raise_exc:
            raise RuntimeError("boom")
        return self.info


def make_session(n_messages: int) -> Session:
    s = Session(system_prompt="sys")
    for i in range(n_messages):
        role = "user" if i % 2 == 0 else "assistant"
        s.add({"role": role, "content": f"mensaje numero {i} con contenido"})
    return s


def test_estimate_tokens_positive():
    assert estimate_tokens([{"role": "user", "content": "hola mundo"}]) > 0


def test_detect_window_from_model_info():
    client = FakeShowClient({"model_info": {"llama.context_length": 4096}})
    assert detect_context_window(client, "m") == 4096


def test_detect_window_missing_key():
    client = FakeShowClient({"model_info": {"llama.embedding_length": 512}})
    assert detect_context_window(client, "m") == DEFAULT_CONTEXT_WINDOW


def test_detect_window_show_fails():
    assert detect_context_window(FakeShowClient(raise_exc=True), "m") == DEFAULT_CONTEXT_WINDOW


def test_no_compact_under_threshold():
    session = make_session(10)
    compactor = Compactor(FakeClient([]), "m", context_window=10_000_000)
    assert compactor.maybe_compact(session) is False
    assert len(session.history) == 10


def test_no_compact_short_history():
    session = make_session(KEEP_RECENT)
    compactor = Compactor(FakeClient([]), "m", context_window=1)
    assert compactor.maybe_compact(session) is False


def test_compacts_with_summary():
    session = make_session(10)
    recent_before = list(session.history[-KEEP_RECENT:])
    notes = []
    client = FakeClient([text_chunks("resumen de lo viejo")])
    compactor = Compactor(client, "m", context_window=1, notify=notes.append)
    assert compactor.maybe_compact(session) is True
    assert len(session.history) == KEEP_RECENT + 1
    assert session.history[0]["role"] == "user"
    assert session.history[0]["content"].startswith("[Resumen de conversación previa]")
    assert "resumen de lo viejo" in session.history[0]["content"]
    assert session.history[1:] == recent_before
    assert any("compacted" in n for n in notes)
    model, messages, tools = client.calls[0]
    assert tools is None
    assert len(messages) == 1


def test_falls_back_to_truncation_on_error():
    class ExplodingClient:
        def chat(self, model, messages, tools=None):
            raise RuntimeError("boom")

    session = make_session(10)
    recent_before = list(session.history[-KEEP_RECENT:])
    notes = []
    compactor = Compactor(ExplodingClient(), "m", context_window=1, notify=notes.append)
    assert compactor.maybe_compact(session) is True
    assert session.history == recent_before
    assert any("truncated" in n for n in notes)


def test_falls_back_on_empty_summary():
    session = make_session(10)
    client = FakeClient([text_chunks("   ")])
    compactor = Compactor(client, "m", context_window=1)
    assert compactor.maybe_compact(session) is True
    assert len(session.history) == KEEP_RECENT
