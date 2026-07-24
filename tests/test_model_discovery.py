from __future__ import annotations

from pathlib import Path

from local_code.model_discovery import (
    discover_models,
    find_models_dir,
    list_local_models,
    ollama_models_candidates,
)


def _make_manifest(models_dir: Path, rel: str) -> None:
    """Create an empty manifest file at manifests/<rel>."""
    p = models_dir / "manifests" / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}")


def test_candidates_env_override_wins(tmp_path):
    env = {"OLLAMA_MODELS": str(tmp_path / "custom")}
    got = ollama_models_candidates("Linux", env, Path("/home/u"))
    assert got == [Path(str(tmp_path / "custom"))]


def test_candidates_macos_default(tmp_path):
    got = ollama_models_candidates("Darwin", {}, tmp_path)
    assert got == [tmp_path / ".ollama" / "models"]


def test_candidates_linux_appends_service_dir(tmp_path):
    got = ollama_models_candidates("Linux", {}, tmp_path)
    assert got == [
        tmp_path / ".ollama" / "models",
        Path("/usr/share/ollama/.ollama/models"),
    ]


def test_candidates_windows_uses_home(tmp_path):
    got = ollama_models_candidates("Windows", {}, tmp_path)
    assert got == [tmp_path / ".ollama" / "models"]


def test_find_models_dir_returns_first_existing(tmp_path):
    home = tmp_path
    (home / ".ollama" / "models").mkdir(parents=True)
    got = find_models_dir(system="Darwin", env={}, home=home)
    assert got == home / ".ollama" / "models"


def test_find_models_dir_none_when_absent(tmp_path):
    got = find_models_dir(system="Darwin", env={}, home=tmp_path)
    assert got is None


def test_list_local_models_reconstructs_names(tmp_path):
    models = tmp_path / "models"
    _make_manifest(models, "registry.ollama.ai/library/llama3.2/latest")
    _make_manifest(models, "registry.ollama.ai/library/qwen2.5-coder/7b")
    _make_manifest(models, "hf.co/user/repo/q4")
    got = list_local_models(models)
    assert got == [
        "hf.co/user/repo:q4",
        "llama3.2:latest",
        "qwen2.5-coder:7b",
    ]


def test_list_local_models_empty_when_no_manifests(tmp_path):
    assert list_local_models(tmp_path) == []


def test_discover_models_end_to_end(tmp_path):
    home = tmp_path
    models = home / ".ollama" / "models"
    _make_manifest(models, "registry.ollama.ai/library/llama3.2/latest")
    models_dir, names = discover_models(system="Darwin", env={}, home=home)
    assert models_dir == models
    assert names == ["llama3.2:latest"]


def test_discover_models_no_dir(tmp_path):
    models_dir, names = discover_models(system="Darwin", env={}, home=tmp_path)
    assert models_dir is None
    assert names == []
