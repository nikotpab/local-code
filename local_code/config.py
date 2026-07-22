from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

CONFIG_PATH = Path.home() / ".local-code" / "config.yaml"


@dataclass
class Config:
    default_model: str = "qwen2.5-coder"
    max_iterations: int = 25
    bash_timeout_seconds: int = 120
    system_prompt: str | None = None
    ollama_host: str = "http://localhost:11434"


def load_config(path: Path | None = None) -> Config:
    path = path or CONFIG_PATH
    if not path.exists():
        return Config()
    data = yaml.safe_load(path.read_text()) or {}
    known = {f.name for f in fields(Config)}
    return Config(**{k: v for k, v in data.items() if k in known})
