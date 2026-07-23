from __future__ import annotations

import json

from local_code.checkpoints import MAX_CHECKPOINTS, CheckpointStore


def make_store(tmp_path):
    return CheckpointStore(dir=tmp_path / "cp")


def test_non_mutating_tool_returns_none(tmp_path):
    assert make_store(tmp_path).snapshot("read_file", {"path": "a"}) is None


def test_missing_path_returns_none(tmp_path):
    assert make_store(tmp_path).snapshot("write_file", {}) is None
    assert make_store(tmp_path).snapshot("write_file", {"path": 42}) is None


def test_snapshot_backs_up_existing_file(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("original")
    store = make_store(tmp_path)
    cp = store.snapshot("edit_file", {"path": str(target)})
    assert cp is not None
    assert cp.backup is not None
    assert (store.dir / f"{cp.id}.bak").read_text() == "original"


def test_snapshot_of_new_file_has_no_backup(tmp_path):
    store = make_store(tmp_path)
    cp = store.snapshot("write_file", {"path": str(tmp_path / "nuevo.txt")})
    assert cp is not None
    assert cp.backup is None


def test_undo_restores_content(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("original")
    store = make_store(tmp_path)
    store.snapshot("edit_file", {"path": str(target)})
    target.write_text("modificado")
    message = store.undo_last()
    assert target.read_text() == "original"
    assert "restaurado" in message


def test_undo_deletes_created_file(tmp_path):
    target = tmp_path / "nuevo.txt"
    store = make_store(tmp_path)
    store.snapshot("write_file", {"path": str(target)})
    target.write_text("creado por el agente")
    message = store.undo_last()
    assert not target.exists()
    assert "eliminado" in message


def test_undo_with_nothing_to_undo(tmp_path):
    assert "nada" in make_store(tmp_path).undo_last().lower()


def test_undo_is_lifo(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("A0")
    b.write_text("B0")
    store = make_store(tmp_path)
    store.snapshot("edit_file", {"path": str(a)})
    a.write_text("A1")
    store.snapshot("edit_file", {"path": str(b)})
    b.write_text("B1")
    store.undo_last()
    assert b.read_text() == "B0"
    assert a.read_text() == "A1"
    store.undo_last()
    assert a.read_text() == "A0"


def test_index_survives_new_instance(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("original")
    store = make_store(tmp_path)
    store.snapshot("edit_file", {"path": str(target)})
    target.write_text("modificado")
    CheckpointStore(dir=store.dir).undo_last()
    assert target.read_text() == "original"


def test_corrupt_index_is_ignored(tmp_path):
    store = make_store(tmp_path)
    (store.dir / "index.json").write_text("{not json")
    assert CheckpointStore(dir=store.dir).list_recent() == []


def test_list_recent_is_newest_first(tmp_path):
    store = make_store(tmp_path)
    for i in range(3):
        f = tmp_path / f"f{i}.txt"
        f.write_text("x")
        store.snapshot("edit_file", {"path": str(f)})
    recent = store.list_recent()
    assert [c.path for c in recent] == [
        str(tmp_path / "f2.txt"),
        str(tmp_path / "f1.txt"),
        str(tmp_path / "f0.txt"),
    ]


def test_retention_trims_oldest(tmp_path):
    store = make_store(tmp_path)
    target = tmp_path / "a.txt"
    target.write_text("x")
    for _ in range(MAX_CHECKPOINTS + 5):
        store.snapshot("edit_file", {"path": str(target)})
    index = json.loads((store.dir / "index.json").read_text())
    assert len(index) == MAX_CHECKPOINTS
    assert len(list(store.dir.glob("*.bak"))) == MAX_CHECKPOINTS


def test_same_second_snapshots_keep_independent_backups(tmp_path, monkeypatch):
    """Two snapshots of the same file in the same wall-clock second with the
    same random token must not clobber each other's backup — undo has to walk
    back through every distinct version."""
    target = tmp_path / "a.txt"
    store = make_store(tmp_path)

    # Freeze both the timestamp and the random token so the OLD id scheme would
    # have produced one shared "{ts}-dead.bak" for both snapshots.
    monkeypatch.setattr(
        "local_code.checkpoints.datetime",
        type(
            "D",
            (),
            {"now": staticmethod(lambda: __import__("datetime").datetime(2026, 7, 23, 12, 0, 0))},
        ),
    )
    monkeypatch.setattr("local_code.checkpoints.secrets.token_hex", lambda n=4: "dead")

    target.write_text("V0")
    store.snapshot("edit_file", {"path": str(target)})
    target.write_text("V1")
    store.snapshot("edit_file", {"path": str(target)})
    target.write_text("V2")

    assert store.undo_last() and target.read_text() == "V1"
    assert store.undo_last() and target.read_text() == "V0"


def test_many_rapid_snapshots_have_unique_backups(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("x")
    store = make_store(tmp_path)
    ids = set()
    for _ in range(300):
        cp = store.snapshot("edit_file", {"path": str(target)})
        assert cp is not None
        ids.add(cp.backup)
    assert len(ids) == 300
