from __future__ import annotations

import json
from collections.abc import Iterator

import requests

from local_code.backends.base import (
    ModelNotFoundError,
    OllamaConnectionError,
    OllamaError,
)


class OllamaClient:
    name = "ollama"

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
        except requests.exceptions.Timeout as e:
            raise OllamaConnectionError(
                f"Timed out waiting for Ollama at {self.host}. Try increasing the "
                "timeout or check that the model is loaded."
            ) from e
        except requests.exceptions.RequestException as e:
            raise OllamaError(f"Request to Ollama at {self.host} failed: {e}") from e
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
        try:
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as e:
                    raise OllamaError(
                        f"Ollama sent an invalid or truncated line while streaming "
                        f"from {self.host}; the connection may have dropped."
                    ) from e
                if "error" in chunk:
                    raise OllamaError(f"Ollama stream error: {chunk['error']}")
                yield chunk
        except requests.exceptions.RequestException as e:
            raise OllamaError(
                f"Stream interrupted while talking to Ollama at {self.host}: {e}"
            ) from e

    def show(self, model: str) -> dict:
        resp = self._post("/api/show", {"model": model}, stream=False, model=model)
        return resp.json()
