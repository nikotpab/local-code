from __future__ import annotations

from local_code.capabilities import CapabilityDetector


class FakeShowClient:
    def __init__(self, info: dict):
        self.info = info
        self.show_calls = 0

    def show(self, model: str) -> dict:
        self.show_calls += 1
        return self.info


def test_supports_tools_true():
    det = CapabilityDetector(FakeShowClient({"capabilities": ["completion", "tools"]}))
    assert det.supports_tools("m") is True


def test_supports_tools_false_when_absent():
    det = CapabilityDetector(FakeShowClient({"capabilities": ["completion"]}))
    assert det.supports_tools("m") is False


def test_supports_tools_false_when_field_missing():
    det = CapabilityDetector(FakeShowClient({"modelfile": "..."}))
    assert det.supports_tools("m") is False


def test_result_cached_per_model():
    client = FakeShowClient({"capabilities": ["tools"]})
    det = CapabilityDetector(client)
    det.supports_tools("m")
    det.supports_tools("m")
    assert client.show_calls == 1
    det.supports_tools("other")
    assert client.show_calls == 2


import json

from local_code.backends.base import OllamaError
from local_code.capabilities import PROBE_TOOL


class FakeProbeClient:
    """A backend with no /api/show, like an OpenAI-compatible server."""

    name = "openai"

    def __init__(self, exc=None):
        self.exc = exc
        self.chat_calls = []

    def show(self, model):
        raise AssertionError("show must not be called for non-ollama backends")

    def chat(self, model, messages, tools=None):
        self.chat_calls.append((model, messages, tools))
        if self.exc is not None:
            raise self.exc
        yield {"message": {"role": "assistant", "content": "hi"}, "done": False}
        yield {"message": {"role": "assistant", "content": ""}, "done": True}


def test_probe_success_means_native(tmp_path):
    client = FakeProbeClient()
    det = CapabilityDetector(client, cache_path=tmp_path / "caps.json")
    assert det.supports_tools("m") is True
    assert client.chat_calls[0][2] == [PROBE_TOOL]


def test_probe_failure_means_react(tmp_path):
    det = CapabilityDetector(
        FakeProbeClient(exc=OllamaError("no tools")), cache_path=tmp_path / "caps.json"
    )
    assert det.supports_tools("m") is False


def test_probe_unexpected_exception_means_react(tmp_path):
    det = CapabilityDetector(
        FakeProbeClient(exc=RuntimeError("weird")), cache_path=tmp_path / "caps.json"
    )
    assert det.supports_tools("m") is False


def test_result_written_to_disk_cache(tmp_path):
    cache = tmp_path / "caps.json"
    CapabilityDetector(FakeProbeClient(), cache_path=cache).supports_tools("m")
    assert json.loads(cache.read_text())["openai:m"] is True


def test_disk_cache_avoids_second_probe(tmp_path):
    cache = tmp_path / "caps.json"
    cache.write_text(json.dumps({"openai:m": True}))
    client = FakeProbeClient(exc=RuntimeError("must not be called"))
    assert CapabilityDetector(client, cache_path=cache).supports_tools("m") is True
    assert client.chat_calls == []


def test_cache_key_includes_backend_and_model(tmp_path):
    cache = tmp_path / "caps.json"
    cache.write_text(json.dumps({"openai:other": True}))
    client = FakeProbeClient(exc=OllamaError("no"))
    assert CapabilityDetector(client, cache_path=cache).supports_tools("m") is False


def test_corrupt_cache_is_ignored(tmp_path):
    cache = tmp_path / "caps.json"
    cache.write_text("{not json")
    assert CapabilityDetector(FakeProbeClient(), cache_path=cache).supports_tools("m") is True


def test_unwritable_cache_does_not_raise(tmp_path):
    det = CapabilityDetector(
        FakeProbeClient(), cache_path=tmp_path / "nope" / "deep" / "caps.json"
    )
    assert det.supports_tools("m") is True


def test_ollama_backend_still_uses_show(tmp_path):
    client = FakeShowClient({"capabilities": ["tools"]})
    client.name = "ollama"
    det = CapabilityDetector(client, cache_path=tmp_path / "caps.json")
    assert det.supports_tools("m") is True
    assert client.show_calls == 1


def test_ollama_backend_never_touches_the_disk_cache(tmp_path):
    """show() is cheap and authoritative, so it is never cached to disk —
    which also keeps detection hermetic for callers that pass no cache path."""
    cache = tmp_path / "caps.json"
    client = FakeShowClient({"capabilities": ["tools"]})
    client.name = "ollama"
    det = CapabilityDetector(client, cache_path=cache)
    assert det.supports_tools("m") is True
    assert not cache.exists()


def test_stale_disk_entry_cannot_override_ollama_show(tmp_path):
    cache = tmp_path / "caps.json"
    cache.write_text(json.dumps({"ollama:m": True}))
    client = FakeShowClient({"capabilities": ["completion"]})
    client.name = "ollama"
    assert CapabilityDetector(client, cache_path=cache).supports_tools("m") is False
