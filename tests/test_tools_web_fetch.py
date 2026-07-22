from __future__ import annotations

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

    def iter_content(self, chunk_size):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]


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
