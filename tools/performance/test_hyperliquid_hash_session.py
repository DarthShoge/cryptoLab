import hashlib
import os
import time

import pytest

from hyperliquid_hash_session import FileHashSession


def test_unchanged_file_is_read_once_per_session(tmp_path):
    path = tmp_path / "payload"
    path.write_bytes(b"original")
    time.sleep(1.05)
    session = FileHashSession()
    expected = hashlib.sha256(b"original").hexdigest()
    assert session(path) == session(path) == expected
    assert (session.misses, session.hits, session.bytes_read) == (1, 1, 8)
    fresh = FileHashSession()
    assert fresh(path) == expected
    assert fresh.bytes_read == 8


def test_same_size_edit_with_restored_mtime_is_rehashed(tmp_path):
    path = tmp_path / "payload"
    path.write_bytes(b"original")
    time.sleep(1.05)
    session = FileHashSession()
    old = session(path)
    before = path.stat()
    path.write_bytes(b"modified")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert session(path) != old
    assert session.misses == 2


def test_atomic_replacement_is_rehashed(tmp_path):
    path, other = tmp_path / "payload", tmp_path / "replacement"
    path.write_bytes(b"original")
    time.sleep(1.05)
    session = FileHashSession()
    old = session(path)
    before = path.stat()
    other.write_bytes(b"modified")
    os.utime(other, ns=(before.st_atime_ns, before.st_mtime_ns))
    other.replace(path)
    assert session(path) != old
    assert session.misses == 2


def test_symlink_substitution_cannot_hit_cache(tmp_path):
    path, other = tmp_path / "payload", tmp_path / "other"
    path.write_bytes(b"original")
    time.sleep(1.05)
    session = FileHashSession()
    session(path)
    path.rename(other)
    path.symlink_to(other)
    with pytest.raises((OSError, ValueError)):
        session(path)


def test_lru_is_bounded(tmp_path):
    paths = [tmp_path / str(i) for i in range(3)]
    session = FileHashSession(max_entries=2)
    for path in paths:
        path.write_bytes(b"data")
    time.sleep(1.05)
    for path in paths:
        session(path)
    session(paths[0])
    assert session.misses == 4
    assert len(session.entries) == 2


def test_mutation_during_hash_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "payload"
    path.write_bytes(b"original")
    session = FileHashSession()
    original = session._digest

    def change(handle):
        result = original(handle)
        path.write_bytes(b"modified")
        return result

    monkeypatch.setattr(session, "_digest", change)
    with pytest.raises(ValueError, match="changed"):
        session(path)
    assert not session.entries


def test_deleted_file_cannot_hit_cache(tmp_path):
    path = tmp_path / "payload"
    path.write_bytes(b"original")
    time.sleep(1.05)
    session = FileHashSession()
    session(path)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        session(path)


def test_recent_write_is_double_checked_and_not_cached(tmp_path):
    path = tmp_path / "payload"
    path.write_bytes(b"original")
    session = FileHashSession()
    assert session(path) == hashlib.sha256(b"original").hexdigest()
    assert session.bytes_read == 16
    assert not session.entries
