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
