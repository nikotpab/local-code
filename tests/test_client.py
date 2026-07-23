from __future__ import annotations

import json

import pytest
import requests

import local_code.backends.ollama as client_mod
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
        # Items that are exceptions are raised mid-iteration instead of
        # yielded, to simulate a connection dropping partway through a
        # stream.
        for item in self._lines:
            if isinstance(item, BaseException):
                raise item
            yield item

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


def test_chat_stream_connection_drop_mid_iteration(monkeypatch):
    # A partial chunk arrives, then the connection drops while iterating
    # iter_lines() itself (outside of the initial requests.post call).
    resp = FakeHTTPResponse(
        lines=[
            chunk_line({"message": {"role": "assistant", "content": "he"}, "done": False}),
            requests.exceptions.ConnectionError("connection dropped mid-stream"),
        ]
    )
    install_post(monkeypatch, response=resp)
    with pytest.raises(OllamaError) as exc_info:
        list(OllamaClient().chat("m", []))
    assert "stream" in str(exc_info.value).lower()


def test_chat_stream_chunked_encoding_error_mid_iteration(monkeypatch):
    resp = FakeHTTPResponse(
        lines=[requests.exceptions.ChunkedEncodingError("connection broken")]
    )
    install_post(monkeypatch, response=resp)
    with pytest.raises(OllamaError):
        list(OllamaClient().chat("m", []))


def test_chat_stream_truncated_json_line(monkeypatch):
    # Connection drops mid-write, leaving a truncated final ndjson line.
    resp = FakeHTTPResponse(lines=[b'{"message": {"role": "assistant", "content": "partial"'])
    install_post(monkeypatch, response=resp)
    with pytest.raises(OllamaError) as exc_info:
        list(OllamaClient().chat("m", []))
    message = str(exc_info.value).lower()
    assert "truncated" in message or "invalid" in message


def test_chat_read_timeout(monkeypatch):
    install_post(monkeypatch, exc=requests.exceptions.ReadTimeout("timed out"))
    with pytest.raises(OllamaConnectionError) as exc_info:
        list(OllamaClient(host="http://x:1").chat("m", []))
    message = str(exc_info.value).lower()
    assert "timed out" in message or "timeout" in message
    assert "http://x:1" in str(exc_info.value)


def test_chat_generic_request_exception(monkeypatch):
    install_post(monkeypatch, exc=requests.exceptions.RequestException("weird failure"))
    with pytest.raises(OllamaError) as exc_info:
        list(OllamaClient().chat("m", []))
    assert type(exc_info.value) is OllamaError


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
