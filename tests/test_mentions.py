from __future__ import annotations

from local_code.mentions import MAX_FILE_CHARS, expand_file_mentions


def test_no_mentions_unchanged():
    assert expand_file_mentions("hola mundo") == ("hola mundo", [])


def test_existing_file_appends_block(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").write_text("x = 1")
    out, warnings = expand_file_mentions("explicá @a.py por favor")
    assert warnings == []
    assert out.startswith("explicá @a.py por favor")
    assert "```a.py\nx = 1\n```" in out


def test_missing_file_warns_text_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out, warnings = expand_file_mentions("mirá @nope.py")
    assert out == "mirá @nope.py"
    assert warnings == ["@nope.py no encontrado"]


def test_trailing_punctuation_stripped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").write_text("x")
    out, warnings = expand_file_mentions("mirá @a.py, y decime")
    assert warnings == []
    assert "```a.py\nx\n```" in out


def test_dedup_same_file_one_block(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").write_text("x")
    out, _ = expand_file_mentions("@a.py y de nuevo @a.py")
    assert out.count("```a.py") == 1


def test_dedup_missing_one_warning(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, warnings = expand_file_mentions("@nope y @nope")
    assert warnings == ["@nope no encontrado"]


def test_truncates_large_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "big.txt").write_text("x" * (MAX_FILE_CHARS + 100))
    out, _ = expand_file_mentions("@big.txt")
    assert "...[truncated]" in out


def test_unreadable_file_warns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "bin.dat").write_bytes(b"\xff\xfe\x00data")
    _, warnings = expand_file_mentions("@bin.dat")
    assert warnings == ["@bin.dat no se pudo leer"]


def test_multiple_files_in_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").write_text("A")
    (tmp_path / "b.py").write_text("B")
    out, _ = expand_file_mentions("@a.py @b.py")
    assert out.index("```a.py") < out.index("```b.py")
