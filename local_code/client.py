from __future__ import annotations

import json
from typing import Iterator

import requests


class OllamaError(Exception):
    pass


class OllamaConnectionError(OllamaError):
    pass


class ModelNotFoundError(OllamaError):
    pass


class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434", timeout: float = 300.0):
        self.host = host.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict, *, stream: bool, model: str):
        try:
            resp = requests.post(
                f"{self.host}{path}", json=payload, stream=stream, timeout=self.timeout
            )
        except requests.exceptions.ConnectionError as e:
            raise OllamaConnectionError(
                f"Cannot reach Ollama at {self.host}. Is it running? Try: ollama serve"
            ) from e
        if resp.status_code == 404:
            raise ModelNotFoundError(
                f"Model '{model}' not found. Try: ollama pull {model}"
            )
        if resp.status_code != 200:
            raise OllamaError(f"Ollama error {resp.status_code}: {resp.text}")
        return resp

    def chat(
        self, model: str, messages: list[dict], tools: list[dict] | None = None
    ) -> Iterator[dict]:
        payload: dict = {"model": model, "messages": messages, "stream": True}
        if tools is not None:
            payload["tools"] = tools
        resp = self._post("/api/chat", payload, stream=True, model=model)
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if "error" in chunk:
                raise OllamaError(f"Ollama stream error: {chunk['error']}")
            yield chunk

    def show(self, model: str) -> dict:
        resp = self._post("/api/show", {"model": model}, stream=False, model=model)
        return resp.json()
