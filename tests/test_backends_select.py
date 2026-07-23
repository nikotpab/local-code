from __future__ import annotations

import pytest

from local_code.backends import select_backend
from local_code.backends.ollama import OllamaClient
from local_code.backends.openai_compat import OpenAICompatClient


def test_auto_ollama_by_default():
    assert isinstance(select_backend("http://localhost:11434"), OllamaClient)


def test_auto_openai_when_host_ends_in_v1():
    assert isinstance(select_backend("http://localhost:1234/v1"), OpenAICompatClient)


def test_auto_openai_ignores_trailing_slash():
    assert isinstance(select_backend("http://localhost:1234/v1/"), OpenAICompatClient)


def test_override_wins_over_url():
    assert isinstance(
        select_backend("http://localhost:1234/v1", override="ollama"), OllamaClient
    )
    assert isinstance(
        select_backend("http://localhost:11434", override="openai"), OpenAICompatClient
    )


def test_invalid_override():
    with pytest.raises(ValueError, match="ollama"):
        select_backend("http://x", override="anthropic")


def test_timeout_and_key_propagate():
    client = select_backend("http://x/v1", timeout=42.0, api_key="k")
    assert client.timeout == 42.0
    assert client.api_key == "k"
