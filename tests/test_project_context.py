from __future__ import annotations

from local_code.project_context import MAX_CONTEXT_CHARS, load_project_context


def test_finds_localcode_md(tmp_path):
    (tmp_path / "LOCALCODE.md").write_text("reglas del proyecto")
    assert load_project_context(tmp_path) == ("LOCALCODE.md", "reglas del proyecto")


def test_falls_back_to_agents_md(tmp_path):
    (tmp_path / "AGENTS.md").write_text("agentes")
    assert load_project_context(tmp_path) == ("AGENTS.md", "agentes")


def test_localcode_has_priority(tmp_path):
    (tmp_path / "LOCALCODE.md").write_text("primero")
    (tmp_path / "AGENTS.md").write_text("segundo")
    assert load_project_context(tmp_path) == ("LOCALCODE.md", "primero")


def test_none_when_absent(tmp_path):
    assert load_project_context(tmp_path) is None


def test_truncates(tmp_path):
    (tmp_path / "LOCALCODE.md").write_text("x" * (MAX_CONTEXT_CHARS + 100))
    result = load_project_context(tmp_path)
    assert result is not None
    _, content = result
    assert content.endswith("...[truncated]")
    assert len(content) < MAX_CONTEXT_CHARS + 50


def test_unreadable_returns_none(tmp_path):
    (tmp_path / "LOCALCODE.md").write_bytes(b"\xff\xfe\x00binario")
    assert load_project_context(tmp_path) is None


def test_default_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "LOCALCODE.md").write_text("desde cwd")
    assert load_project_context() == ("LOCALCODE.md", "desde cwd")
