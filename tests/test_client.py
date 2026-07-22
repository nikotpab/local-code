from __future__ import annotations

import json

import pytest
import requests

import local_code.client as client_mod
from local_code.client import (
    ModelNotFoundError,
    OllamaClient,
    OllamaConnectionError,
    OllamaError,
)


class FakeHTTPResponse:
    def __init__(self, status_code=200, lines=None, json_data=None, text=""):
        self.status_code = status_code
        self._lines = lines or []
        self._json = json_data
        self.text = text

    def iter_lines(self):
        return iter(self._lines)

    def json(self):
        return self._json


def install_post(monkeypatch, response=None, exc=None):
    """Replace requests.post inside client module; record calls."""
    calls = []

    def fake_post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        if exc is not None:
            raise exc
        return response

    monkeypatch.setattr(client_mod.requests, "post", fake_post)
    return calls


def chunk_line(payload: dict) -> bytes:
    return json.dumps(payload).encode()


def test_chat_streams_chunks_and_builds_payload(monkeypatch):
    resp = FakeHTTPResponse(
        lines=[
            chunk_line({"message": {"role": "assistant", "content": "ho"}, "done": False}),
            chunk_line({"message": {"role": "assistant", "content": "la"}, "done": True}),
        ]
    )
    calls = install_post(monkeypatch, response=resp)
    c = OllamaClient(host="http://x:1")
    chunks = list(c.chat("m1", [{"role": "user", "content": "hi"}]))
    assert [ch["message"]["content"] for ch in chunks] == ["ho", "la"]
    assert calls[0]["url"] == "http://x:1/api/chat"
    payload = calls[0]["json"]
    assert payload["model"] == "m1"
    assert payload["stream"] is True
    assert "tools" not in payload


def test_chat_includes_tools_when_given(monkeypatch):
    resp = FakeHTTPResponse(lines=[])
    calls = install_post(monkeypatch, response=resp)
    schemas = [{"type": "function", "function": {"name": "t"}}]
    list(OllamaClient().chat("m", [], tools=schemas))
    assert calls[0]["json"]["tools"] == schemas


def test_chat_connection_error(monkeypatch):
    install_post(monkeypatch, exc=requests.exceptions.ConnectionError("refused"))
    with pytest.raises(OllamaConnectionError, match="ollama serve"):
        list(OllamaClient().chat("m", []))


def test_chat_model_not_found(monkeypatch):
    install_post(monkeypatch, response=FakeHTTPResponse(status_code=404))
    with pytest.raises(ModelNotFoundError, match="ollama pull m"):
        list(OllamaClient().chat("m", []))


def test_chat_other_http_error(monkeypatch):
    install_post(monkeypatch, response=FakeHTTPResponse(status_code=500, text="boom"))
    with pytest.raises(OllamaError, match="500"):
        list(OllamaClient().chat("m", []))


def test_chat_midstream_error_chunk(monkeypatch):
    resp = FakeHTTPResponse(lines=[chunk_line({"error": "out of memory"})])
    install_post(monkeypatch, response=resp)
    with pytest.raises(OllamaError, match="out of memory"):
        list(OllamaClient().chat("m", []))


def test_show_success(monkeypatch):
    resp = FakeHTTPResponse(json_data={"capabilities": ["completion", "tools"]})
    calls = install_post(monkeypatch, response=resp)
    info = OllamaClient(host="http://x:1").show("m1")
    assert info["capabilities"] == ["completion", "tools"]
    assert calls[0]["url"] == "http://x:1/api/show"
    assert calls[0]["json"] == {"model": "m1"}


def test_show_model_not_found(monkeypatch):
    install_post(monkeypatch, response=FakeHTTPResponse(status_code=404))
    with pytest.raises(ModelNotFoundError):
        OllamaClient().show("m")


def test_show_connection_error(monkeypatch):
    install_post(monkeypatch, exc=requests.exceptions.ConnectionError("refused"))
    with pytest.raises(OllamaConnectionError):
        OllamaClient().show("m")
