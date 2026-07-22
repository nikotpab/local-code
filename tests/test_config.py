from __future__ import annotations

from local_code.config import Config, load_config


def test_missing_file_returns_defaults(tmp_path):
    cfg = load_config(tmp_path / "nope.yaml")
    assert cfg == Config()
    assert cfg.default_model == "qwen2.5-coder"
    assert cfg.max_iterations == 25
    assert cfg.bash_timeout_seconds == 120
    assert cfg.system_prompt is None
    assert cfg.ollama_host == "http://localhost:11434"


def test_full_file(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "default_model: llama3.1\n"
        "max_iterations: 10\n"
        "bash_timeout_seconds: 30\n"
        "system_prompt: sos un experto\n"
        "ollama_host: http://192.168.1.5:11434\n"
    )
    cfg = load_config(p)
    assert cfg.default_model == "llama3.1"
    assert cfg.max_iterations == 10
    assert cfg.bash_timeout_seconds == 30
    assert cfg.system_prompt == "sos un experto"
    assert cfg.ollama_host == "http://192.168.1.5:11434"


def test_partial_file_merges_defaults(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("default_model: mistral\n")
    cfg = load_config(p)
    assert cfg.default_model == "mistral"
    assert cfg.max_iterations == 25


def test_unknown_keys_ignored(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("default_model: mistral\nfuture_option: true\n")
    cfg = load_config(p)
    assert cfg.default_model == "mistral"


def test_empty_file(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("")
    assert load_config(p) == Config()
