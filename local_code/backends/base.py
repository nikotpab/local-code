from __future__ import annotations


class OllamaError(Exception):
    pass


class OllamaConnectionError(OllamaError):
    pass


class ModelNotFoundError(OllamaError):
    pass


BackendError = OllamaError
