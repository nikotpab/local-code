"""Discover Ollama models on disk, OS-aware.

Ollama stores each model as a manifest file under::

    <models>/manifests/<registry>/<namespace>/<model>/<tag>

plus content-addressed blobs under ``<models>/blobs``. This module locates the
``<models>`` directory (respecting the ``OLLAMA_MODELS`` override and falling
back to per-OS defaults) and reconstructs the ``name:tag`` strings from the
manifest tree — the same names ``ollama list`` and the ``--model`` flag use.
"""

from __future__ import annotations

import os
import platform
from collections.abc import Mapping
from pathlib import Path

# Default registry/namespace that Ollama hides from display names:
# `registry.ollama.ai/library/llama3.2:latest` shows as `llama3.2:latest`.
_DEFAULT_REGISTRY = "registry.ollama.ai"
_DEFAULT_NAMESPACE = "library"


def ollama_models_candidates(
    system: str,
    env: Mapping[str, str],
    home: Path,
) -> list[Path]:
    """Return candidate models directories in priority order.

    ``OLLAMA_MODELS`` wins outright when set. Otherwise the per-user default
    ``~/.ollama/models`` comes first; on Linux the system-service location is
    appended as a fallback.
    """
    override = env.get("OLLAMA_MODELS")
    if override:
        return [Path(override)]

    candidates = [home / ".ollama" / "models"]
    if system == "Linux":
        candidates.append(Path("/usr/share/ollama/.ollama/models"))
    return candidates


def find_models_dir(
    system: str | None = None,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path | None:
    """Return the first existing models directory, or None if none exist."""
    system = system if system is not None else platform.system()
    env = env if env is not None else os.environ
    home = home if home is not None else Path.home()
    for candidate in ollama_models_candidates(system, env, home):
        if candidate.is_dir():
            return candidate
    return None


def _manifest_to_name(rel_parts: tuple[str, ...]) -> str | None:
    """Turn a manifest path relative to ``manifests/`` into a ``name:tag``.

    ``rel_parts`` is like ``(registry, namespace, model, tag)``; the last
    element is the tag and the one before it the model. Leading default
    registry/namespace segments are dropped for a clean display name.
    """
    if len(rel_parts) < 2:
        return None
    tag = rel_parts[-1]
    model = rel_parts[-2]
    prefix = list(rel_parts[:-2])
    if prefix and prefix[0] == _DEFAULT_REGISTRY:
        prefix = prefix[1:]
    if prefix and prefix[0] == _DEFAULT_NAMESPACE:
        prefix = prefix[1:]
    segments = prefix + [model]
    return f"{'/'.join(segments)}:{tag}"


def list_local_models(models_dir: Path) -> list[str]:
    """List installed model names by scanning the manifest tree.

    Returns a sorted, de-duplicated list of ``name:tag`` strings. A missing or
    empty ``manifests`` directory yields an empty list.
    """
    manifests = models_dir / "manifests"
    if not manifests.is_dir():
        return []
    names: set[str] = set()
    for path in manifests.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(manifests).parts
        name = _manifest_to_name(rel_parts)
        if name is not None:
            names.add(name)
    return sorted(names)


def discover_models(
    system: str | None = None,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> tuple[Path | None, list[str]]:
    """Locate the models directory and list the models inside it.

    Returns ``(models_dir, names)``. ``models_dir`` is None when no directory
    was found; ``names`` is empty in that case too.
    """
    models_dir = find_models_dir(system=system, env=env, home=home)
    if models_dir is None:
        return None, []
    return models_dir, list_local_models(models_dir)
