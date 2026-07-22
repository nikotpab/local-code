from __future__ import annotations

import time
from html.parser import HTMLParser

import requests

from local_code.tools.context import ToolContext

NAME = "web_fetch"
DESCRIPTION = "Fetch an http(s) URL and return its text content. HTML is stripped to plain text."
PARAMETERS = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "http(s) URL to fetch"}
    },
    "required": ["url"],
}
REQUIRES_CONFIRMATION = True
MAX_BYTES = 5_000_000
MAX_CHARS = 50_000
TIMEOUT = 30


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.parts.append(data.strip())


def _extract_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return "\n".join(parser.parts)


def run(arguments: dict, context: ToolContext) -> str:
    url = arguments.get("url")
    if not isinstance(url, str) or not url:
        return "Error: url is required"
    if not url.lower().startswith(("http://", "https://")):
        return "Error: only http(s) URLs are supported"
    deadline = time.monotonic() + TIMEOUT
    try:
        resp = requests.get(url, timeout=TIMEOUT, stream=True)
    except requests.RequestException as e:
        return f"Error: {e}"
    if resp.status_code != 200:
        return f"Error: HTTP {resp.status_code}"
    raw = b""
    timed_out_empty = False
    try:
        for chunk in resp.iter_content(65536):
            raw += chunk
            if len(raw) >= MAX_BYTES:
                break
            if time.monotonic() > deadline:
                if not raw:
                    timed_out_empty = True
                break
    except requests.RequestException as e:
        return f"Error: {e}"
    finally:
        resp.close()
    if timed_out_empty:
        return f"Error: timed out after {TIMEOUT}s"
    try:
        text = raw.decode(resp.encoding or "utf-8", errors="replace")
    except LookupError:
        text = raw.decode("utf-8", errors="replace")
    if "text/html" in resp.headers.get("Content-Type", ""):
        text = _extract_text(text)
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + "\n...[truncated]"
    return text


def preview(arguments: dict) -> str:
    return f"GET {arguments.get('url', '')}"
