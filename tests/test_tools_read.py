from __future__ import annotations

import pytest

from local_code.tools import list_dir, read_file
from local_code.tools.context import ToolContext

CTX = ToolContext()


def test_tool_module_contract():
    for mod in (read_file, list_dir):
        assert isinstance(mod.NAME, str)
        assert isinstance(mod.DESCRIPTION, str)
        assert mod.PARAMETERS["type"] == "object"
        assert mod.REQUIRES_CONFIRMATION is False


def test_read_file(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hola\nmundo\n")
    assert read_file.run({"path": str(f)}, CTX) == "hola\nmundo\n"


def test_read_file_truncates(tmp_path):
    f = tmp_path / "big.txt"
    f.write_text("x" * (read_file.MAX_CHARS + 10))
    out = read_file.run({"path": str(f)}, CTX)
    assert out.endswith("...[truncated]")
    assert len(out) < read_file.MAX_CHARS + 100


def test_read_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_file.run({"path": str(tmp_path / "nope.txt")}, CTX)


def test_list_dir(tmp_path):
    (tmp_path / "b.txt").write_text("")
    (tmp_path / "a_dir").mkdir()
    out = list_dir.run({"path": str(tmp_path)}, CTX)
    assert out == "a_dir/\nb.txt"


def test_list_dir_default_is_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "only.txt").write_text("")
    assert list_dir.run({}, CTX) == "only.txt"


def test_list_dir_empty(tmp_path):
    assert list_dir.run({"path": str(tmp_path)}, CTX) == "(empty directory)"
