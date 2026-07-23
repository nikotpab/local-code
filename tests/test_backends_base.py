from __future__ import annotations

from local_code import client as client_module
from local_code.backends.base import (
    BackendError,
    ModelNotFoundError,
    OllamaConnectionError,
    OllamaError,
)
from local_code.backends.ollama import OllamaClient


def test_exception_hierarchy():
    assert issubclass(OllamaConnectionError, OllamaError)
    assert issubclass(ModelNotFoundError, OllamaError)
    assert BackendError is OllamaError


def test_client_module_still_exports_everything():
    assert client_module.OllamaClient is OllamaClient
    assert client_module.OllamaError is OllamaError
    assert client_module.OllamaConnectionError is OllamaConnectionError
    assert client_module.ModelNotFoundError is ModelNotFoundError


def test_backend_has_name():
    assert OllamaClient.name == "ollama"
    assert OllamaClient(host="http://x:1").name == "ollama"
