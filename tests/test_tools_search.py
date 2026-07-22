from __future__ import annotations

import re

import pytest

from local_code.tools import glob as glob_tool
from local_code.tools import grep
from local_code.tools.context import ToolContext

CTX = ToolContext()


def test_glob_matches_sorted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "b.py").write_text("")
    (tmp_path / "a.py").write_text("")
    (tmp_path / "c.txt").write_text("")
    assert glob_tool.run({"pattern": "*.py"}, CTX) == "a.py\nb.py"


def test_glob_recursive(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "m.py").write_text("")
    out = glob_tool.run({"pattern": "**/*.py"}, CTX)
    assert "pkg/m.py" in out


def test_glob_no_matches(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert glob_tool.run({"pattern": "*.rs"}, CTX) == "(no matches)"


def test_glob_truncates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for i in range(glob_tool.MAX_RESULTS + 5):
        (tmp_path / f"f{i:04d}.txt").write_text("")
    out = glob_tool.run({"pattern": "*.txt"}, CTX)
    lines = out.splitlines()
    assert lines[-1] == "...[truncated]"
    assert len(lines) == glob_tool.MAX_RESULTS + 1


def test_grep_finds_matches(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\ndef foo():\n    pass\n")
    (tmp_path / "b.py").write_text("def bar():\n    pass\n")
    out = grep.run({"pattern": r"def \w+", "path": str(tmp_path)}, CTX)
    assert f"{tmp_path}/a.py:2: def foo():" in out
    assert f"{tmp_path}/b.py:1: def bar():" in out


def test_grep_skips_git_dir(tmp_path):
    git = tmp_path / ".git"
    git.mkdir()
    (git / "config").write_text("match_me\n")
    (tmp_path / "real.txt").write_text("match_me\n")
    out = grep.run({"pattern": "match_me", "path": str(tmp_path)}, CTX)
    assert ".git" not in out
    assert "real.txt" in out


def test_grep_skips_binary(tmp_path):
    (tmp_path / "bin.dat").write_bytes(b"\xff\xfe\x00match")
    (tmp_path / "ok.txt").write_text("match\n")
    out = grep.run({"pattern": "match", "path": str(tmp_path)}, CTX)
    assert "ok.txt" in out
    assert "bin.dat" not in out


def test_grep_no_matches(tmp_path):
    (tmp_path / "a.txt").write_text("nothing here\n")
    assert grep.run({"pattern": "zzz", "path": str(tmp_path)}, CTX) == "(no matches)"


def test_grep_invalid_regex_raises(tmp_path):
    with pytest.raises(re.error):
        grep.run({"pattern": "(", "path": str(tmp_path)}, CTX)
