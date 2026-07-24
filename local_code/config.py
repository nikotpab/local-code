from __future__ import annotations

import json
import logging
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

CONFIG_PATH = Path.home() / ".local-code" / "config.yaml"
MCP_CONFIG_PATH = Path.home() / ".local-code" / "mcp.json"

logger = logging.getLogger(__name__)


@dataclass
class Config:
    default_model: str = "qwen2.5-coder"
    max_iterations: int = 25
    bash_timeout_seconds: int = 120
    system_prompt: str | None = None
    ollama_host: str = "http://localhost:11434"
    context_window: int | None = None
    backend: str | None = None
    api_key: str | None = None


def load_config(path: Path | None = None) -> Config:
    path = path or CONFIG_PATH
    if not path.exists():
        return Config()
    data = yaml.safe_load(path.read_text()) or {}
    known = {f.name for f in fields(Config)}
    return Config(**{k: v for k, v in data.items() if k in known})


# Expected python types per config key, used by validate_config. Kept explicit
# (rather than reflecting dataclass annotations) because __future__ annotations
# turns field.type into strings.
_EXPECTED_TYPES: dict[str, tuple[type, ...]] = {
    "default_model": (str,),
    "max_iterations": (int,),
    "bash_timeout_seconds": (int,),
    "system_prompt": (str, type(None)),
    "ollama_host": (str,),
    "context_window": (int, type(None)),
    "backend": (str, type(None)),
    "api_key": (str, type(None)),
}


def validate_config(path: Path | None = None) -> list[str]:
    """Check the config file and return a list of problem strings.

    An empty list means valid. A missing file is valid (all defaults apply).
    """
    path = path or CONFIG_PATH
    if not path.exists():
        return []
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        return [f"YAML parse error: {exc}"]
    if data is None:
        return []
    if not isinstance(data, dict):
        return ["top-level config must be a mapping (key: value pairs)"]

    problems: list[str] = []
    for key, value in data.items():
        if key not in _EXPECTED_TYPES:
            problems.append(f"unknown key: {key!r}")
            continue
        expected = _EXPECTED_TYPES[key]
        # bool is an int subclass in Python; reject it where an int is wanted.
        if isinstance(value, bool) and int in expected and bool not in expected:
            problems.append(f"{key}: expected {_type_names(expected)}, got bool")
            continue
        if not isinstance(value, expected):
            problems.append(f"{key}: expected {_type_names(expected)}, got {type(value).__name__}")

    backend = data.get("backend")
    if backend is not None and backend not in ("ollama", "openai"):
        problems.append(f"backend: must be 'ollama' or 'openai', got {backend!r}")

    return problems


def _type_names(types: tuple[type, ...]) -> str:
    return " or ".join("null" if t is type(None) else t.__name__ for t in types)


def load_mcp_server_configs(path: Path | None = None) -> list[dict]:
    """Load MCP server configurations from ``~/.local-code/mcp.json``.

    The file uses the same shape as standard MCP client config:

    .. code-block:: json

        {
          "mcpServers": {
            "filesystem": {
              "command": "npx",
              "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
              "env": {},
              "trust": false
            }
          }
        }

    Returns a list of flat config dicts, each with a ``"name"`` key injected
    from the map key.  On any error (missing file, bad JSON, wrong shape) returns
    an empty list so the CLI continues normally.
    """
    path = path or MCP_CONFIG_PATH
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("mcp.json could not be read: %s", exc)
        return []
    if not isinstance(raw, dict):
        logger.warning("mcp.json: expected a JSON object at the top level")
        return []
    servers = raw.get("mcpServers")
    if not isinstance(servers, dict):
        return []
    configs: list[dict] = []
    for name, cfg in servers.items():
        if not isinstance(cfg, dict):
            logger.warning("mcp.json: server '%s' is not an object, skipping", name)
            continue
        entry = dict(cfg)
        entry["name"] = name
        configs.append(entry)
    return configs
