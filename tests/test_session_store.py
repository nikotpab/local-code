from __future__ import annotations

import json
import os
import re

import pytest

from local_code.session_store import SessionNotFoundError, SessionStore


def make_store(tmp_path):
    return SessionStore(dir=tmp_path / "sessions")


def test_new_id_format(tmp_path):
    assert re.fullmatch(r"\d{8}-\d{6}", make_store(tmp_path).new_id())


def test_save_load_roundtrip(tmp_path):
    store = make_store(tmp_path)
    history = [{"role": "user", "content": "hola"}, {"role": "assistant", "content": "hey"}]
    store.save("s1", "qwen", "sys", history)
    data = store.load("s1")
    assert data["id"] == "s1"
    assert data["model"] == "qwen"
    assert data["system_prompt"] == "sys"
    assert data["history"] == history
    assert data["created_at"] and data["updated_at"]
    assert data["cwd"]


def test_save_preserves_created_at(tmp_path):
    store = make_store(tmp_path)
    store.save("s1", "m", None, [])
    first = store.load("s1")["created_at"]
    store.save("s1", "m", None, [{"role": "user", "content": "x"}])
    assert store.load("s1")["created_at"] == first


def test_save_leaves_no_tmp_file(tmp_path):
    store = make_store(tmp_path)
    store.save("s1", "m", None, [])
    assert [p.name for p in store.dir.iterdir()] == ["s1.json"]


def test_load_missing_raises(tmp_path):
    with pytest.raises(SessionNotFoundError, match="nope"):
        make_store(tmp_path).load("nope")


def test_load_corrupt_raises(tmp_path):
    store = make_store(tmp_path)
    (store.dir / "bad.json").write_text("{no es json")
    with pytest.raises(SessionNotFoundError):
        store.load("bad")


def test_latest_id(tmp_path):
    store = make_store(tmp_path)
    assert store.latest_id() is None
    store.save("a", "m", None, [])
    store.save("b", "m", None, [])
    os.utime(store.dir / "a.json", (1000, 1000))
    os.utime(store.dir / "b.json", (2000, 2000))
    assert store.latest_id() == "b"


def test_list_sessions_order_and_fields(tmp_path):
    store = make_store(tmp_path)
    store.save("a", "m1", None, [{"role": "user", "content": "primera pregunta " + "x" * 100}])
    store.save("b", "m2", None, [{"role": "assistant", "content": "sin user"}])
    os.utime(store.dir / "a.json", (1000, 1000))
    os.utime(store.dir / "b.json", (2000, 2000))
    sessions = store.list_sessions()
    assert [s["id"] for s in sessions] == ["b", "a"]
    assert sessions[0]["model"] == "m2"
    assert sessions[0]["first_message"] == ""
    assert sessions[1]["first_message"].startswith("primera pregunta")
    assert len(sessions[1]["first_message"]) <= 60


def test_list_sessions_skips_corrupt_and_limits(tmp_path):
    store = make_store(tmp_path)
    (store.dir / "bad.json").write_text("{")
    for i in range(12):
        store.save(f"s{i:02d}", "m", None, [])
    sessions = store.list_sessions(limit=5)
    assert len(sessions) == 5
    assert all(s["id"].startswith("s") for s in sessions)
