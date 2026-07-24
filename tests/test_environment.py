from __future__ import annotations

from pathlib import Path

from local_code.environment import (
    MAX_HOME_ENTRIES,
    environment_block,
    git_repo_root,
    home_directories,
)


def test_git_root_in_cwd(tmp_path):
    (tmp_path / ".git").mkdir()
    assert git_repo_root(tmp_path) == tmp_path


def test_git_root_in_ancestor(tmp_path):
    (tmp_path / ".git").mkdir()
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    assert git_repo_root(deep) == tmp_path


def test_git_root_none(tmp_path):
    assert git_repo_root(tmp_path) is None


def test_home_directories_sorted_visible_dirs(tmp_path):
    (tmp_path / "Desktop").mkdir()
    (tmp_path / "Documents").mkdir()
    (tmp_path / ".config").mkdir()
    (tmp_path / "notes.txt").write_text("x")
    assert home_directories(tmp_path) == ["Desktop", "Documents"]


def test_home_directories_caps_with_ellipsis(tmp_path):
    for i in range(MAX_HOME_ENTRIES + 5):
        (tmp_path / f"dir{i:03d}").mkdir()
    result = home_directories(tmp_path)
    assert len(result) == MAX_HOME_ENTRIES + 1
    assert result[-1] == "…"


def test_home_directories_unreadable_returns_empty(tmp_path):
    missing = tmp_path / "nope"
    assert home_directories(missing) == []


def test_environment_block_has_core_facts(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / "Desktop").mkdir()
    cwd = home / "proyecto"
    cwd.mkdir()
    block = environment_block(cwd=cwd, home=home)
    assert block.startswith("# Environment")
    assert f"Home directory: {home}" in block
    assert f"Working directory: {cwd}" in block
    assert "Operating system:" in block
    assert "Today:" in block
    assert "do not translate or guess folder names" in block
    assert "Desktop" in block


def test_environment_block_omits_folders_when_none(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    block = environment_block(cwd=home, home=home)
    # The folders paragraph (and only it) carries this instruction.
    assert "do not translate or guess folder names" not in block


def test_environment_block_reports_git_repo(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    repo = home / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    block = environment_block(cwd=repo, home=home)
    assert f"Git repository: {repo}" in block


def test_environment_block_reports_no_git(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    block = environment_block(cwd=home, home=home)
    assert "not a git repository" in block


def test_environment_block_never_raises(monkeypatch, tmp_path):
    import local_code.environment as env

    def boom():
        raise RuntimeError("no user")

    monkeypatch.setattr(env.getpass, "getuser", boom)
    monkeypatch.delenv("USER", raising=False)
    block = environment_block(cwd=tmp_path, home=tmp_path)
    assert "User: unknown" in block
