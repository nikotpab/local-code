from __future__ import annotations

from local_code.backends.base import (
    BackendError,
    ModelNotFoundError,
    OllamaConnectionError,
    OllamaError,
)
from local_code.backends.ollama import OllamaClient
from local_code.backends.openai_compat import OpenAICompatClient

__all__ = [
    "BackendError",
    "ModelNotFoundError",
    "OllamaClient",
    "OllamaConnectionError",
    "OllamaError",
    "OpenAICompatClient",
    "select_backend",
]

BACKENDS = ("ollama", "openai")


def select_backend(
    host: str,
    override: str | None = None,
    timeout: float = 300.0,
    api_key: str | None = None,
):
    if override is not None and override not in BACKENDS:
        raise ValueError(
            f"Unknown backend '{override}'. Valid options: {', '.join(BACKENDS)}"
        )
    kind = override
    if kind is None:
        kind = "openai" if host.rstrip("/").endswith("/v1") else "ollama"
    if kind == "openai":
        return OpenAICompatClient(host=host, timeout=timeout, api_key=api_key)
    return OllamaClient(host=host, timeout=timeout)
