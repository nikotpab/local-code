from __future__ import annotations

import io

from rich.console import Console

from local_code.cli import run_config_command
from local_code.config import validate_config


def _console():
    buf = io.StringIO()
    return Console(file=buf, force_terminal=False), buf


def test_validate_missing_file_is_valid(tmp_path):
    assert validate_config(tmp_path / "nope.yaml") == []


def test_validate_unknown_key(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("default_model: qwen\nbogus: 1\n")
    problems = validate_config(p)
    assert any("unknown key" in x and "bogus" in x for x in problems)


def test_validate_type_mismatch(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("max_iterations: not-a-number\n")
    problems = validate_config(p)
    assert any("max_iterations" in x for x in problems)


def test_validate_bad_backend(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("backend: azure\n")
    problems = validate_config(p)
    assert any("backend" in x for x in problems)


def test_validate_bool_rejected_for_int(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("max_iterations: true\n")
    problems = validate_config(p)
    assert any("max_iterations" in x and "bool" in x for x in problems)


def test_validate_yaml_error(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("key: [unclosed\n")
    problems = validate_config(p)
    assert problems and "YAML" in problems[0]


def test_config_command_path():
    console, buf = _console()
    assert run_config_command(["path"], console) == 0
    assert ".local-code" in buf.getvalue()


def test_config_command_show_redacts_api_key(tmp_path, monkeypatch):
    import local_code.cli as cli
    import local_code.config as config

    p = tmp_path / "config.yaml"
    p.write_text("api_key: super-secret-value\n")
    monkeypatch.setattr(config, "CONFIG_PATH", p)
    monkeypatch.setattr(cli, "CONFIG_PATH", p)

    console, buf = _console()
    assert run_config_command(["show"], console) == 0
    out = buf.getvalue()
    assert "super-secret-value" not in out
    assert "redacted" in out


def test_config_command_unknown_action():
    console, buf = _console()
    assert run_config_command(["bogus"], console) == 2
    assert "unknown config action" in buf.getvalue()
