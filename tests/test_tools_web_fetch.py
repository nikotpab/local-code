from __future__ import annotations

import itertools

import requests

import local_code.tools.web_fetch as web_fetch
from local_code.tools.context import ToolContext

CTX = ToolContext()


class FakeResponse:
    def __init__(self, status_code=200, body=b"", content_type="text/plain", encoding="utf-8"):
        self.status_code = status_code
        self._body = body
        self.headers = {"Content-Type": content_type}
        self.encoding = encoding
        self.closed = False

    def iter_content(self, chunk_size):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]

    def close(self):
        self.closed = True


def install_get(monkeypatch, response=None, exc=None):
    def fake_get(url, **kwargs):
        if exc is not None:
            raise exc
        return response

    monkeypatch.setattr(web_fetch.requests, "get", fake_get)


def test_plain_text(monkeypatch):
    install_get(monkeypatch, FakeResponse(body=b"hola mundo"))
    assert web_fetch.run({"url": "https://x.test/a"}, CTX) == "hola mundo"


def test_html_stripped(monkeypatch):
    html = b"<html><head><script>var x=1;</script><style>.a{}</style></head><body><h1>Titulo</h1><p>parrafo</p></body></html>"
    install_get(monkeypatch, FakeResponse(body=html, content_type="text/html; charset=utf-8"))
    out = web_fetch.run({"url": "https://x.test"}, CTX)
    assert "Titulo" in out
    assert "parrafo" in out
    assert "var x=1" not in out
    assert ".a{}" not in out


def test_http_error(monkeypatch):
    install_get(monkeypatch, FakeResponse(status_code=404))
    assert web_fetch.run({"url": "https://x.test"}, CTX) == "Error: HTTP 404"


def test_bad_scheme():
    out = web_fetch.run({"url": "ftp://x.test"}, CTX)
    assert out == "Error: only http(s) URLs are supported"


def test_request_exception(monkeypatch):
    install_get(monkeypatch, exc=requests.exceptions.ConnectionError("refused"))
    out = web_fetch.run({"url": "https://x.test"}, CTX)
    assert out.startswith("Error:")


def test_truncates_chars(monkeypatch):
    install_get(monkeypatch, FakeResponse(body=b"x" * (web_fetch.MAX_CHARS + 100)))
    out = web_fetch.run({"url": "https://x.test"}, CTX)
    assert out.endswith("...[truncated]")
    assert len(out) < web_fetch.MAX_CHARS + 50


def test_preview_and_contract():
    assert web_fetch.preview({"url": "https://x.test"}) == "GET https://x.test"
    assert web_fetch.REQUIRES_CONFIRMATION is True


def test_missing_url():
    assert web_fetch.run({}, CTX) == "Error: url is required"


def test_non_string_url():
    assert web_fetch.run({"url": None}, CTX) == "Error: url is required"


def test_preview_missing_url():
    assert web_fetch.preview({}) == "GET "


def test_uppercase_scheme_accepted(monkeypatch):
    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        return FakeResponse(body=b"ok")

    monkeypatch.setattr(web_fetch.requests, "get", fake_get)
    out = web_fetch.run({"url": "HTTP://X.test/Path"}, CTX)
    assert out == "ok"
    # original casing must reach the actual request, only the scheme check is lowercased
    assert captured["url"] == "HTTP://X.test/Path"


def test_bogus_charset_falls_back_to_utf8(monkeypatch):
    body = "café".encode()
    install_get(monkeypatch, FakeResponse(body=body, encoding="bogus-charset"))
    out = web_fetch.run({"url": "https://x.test"}, CTX)
    assert out == "café"


def test_stream_error_mid_body(monkeypatch):
    def gen(chunk_size):
        yield b"partial"
        raise requests.exceptions.ChunkedEncodingError("connection broken")

    resp = FakeResponse(body=b"unused")
    resp.iter_content = gen
    install_get(monkeypatch, resp)
    out = web_fetch.run({"url": "https://x.test"}, CTX)
    assert out.startswith("Error:")
    assert resp.closed is True


def test_stalls_past_deadline_returns_partial(monkeypatch):
    # Fake clock: first call sets the deadline, second call (after the first
    # chunk arrives) already reads far past it. The generator would raise if
    # asked for a second chunk, proving the loop stopped rather than hanging.
    clock = itertools.count(0, 100)
    monkeypatch.setattr(web_fetch.time, "monotonic", lambda: next(clock))

    def gen(chunk_size):
        yield b"partial-data"
        raise AssertionError("iter_content consumed past the wall-clock deadline")

    resp = FakeResponse(body=b"unused")
    resp.iter_content = gen
    install_get(monkeypatch, resp)
    out = web_fetch.run({"url": "https://x.test"}, CTX)
    assert out == "partial-data"
    assert resp.closed is True


def test_timeout_with_no_bytes_received(monkeypatch):
    clock = itertools.count(0, 100)
    monkeypatch.setattr(web_fetch.time, "monotonic", lambda: next(clock))

    def gen(chunk_size):
        yield b""
        raise AssertionError("iter_content consumed past the wall-clock deadline")

    resp = FakeResponse(body=b"unused")
    resp.iter_content = gen
    install_get(monkeypatch, resp)
    out = web_fetch.run({"url": "https://x.test"}, CTX)
    assert out == f"Error: timed out after {web_fetch.TIMEOUT}s"
    assert resp.closed is True


def test_closes_response_after_read(monkeypatch):
    resp = FakeResponse(body=b"hola")
    install_get(monkeypatch, resp)
    web_fetch.run({"url": "https://x.test"}, CTX)
    assert resp.closed is True
