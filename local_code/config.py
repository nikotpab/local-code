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
