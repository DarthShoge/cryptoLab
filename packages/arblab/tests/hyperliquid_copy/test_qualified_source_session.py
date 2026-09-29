import json
from pathlib import Path

import pytest

from .test_qualified_day import qualified


def first_file(pin):
    return Path(json.loads(Path(pin["path"]).read_text())["files"][0]["path"])


@pytest.mark.parametrize("fault", ["alias_write", "new_link"])
def test_existing_canonical_hardlink_is_verified_then_guarded(
    qualified, tmp_path, fault
):
    from arblab.hyperliquid_copy.qualified_source_session import QualifiedSourceSession

    path = first_file(qualified)
    alias = tmp_path / "registered-copy.parquet"
    alias.hardlink_to(path)
    session = QualifiedSourceSession(qualified)
    session.verify()
    if fault == "alias_write":
        with alias.open("ab") as handle:
            handle.write(b"changed through registered alias")
    else:
        (tmp_path / "additional-copy.parquet").hardlink_to(path)
    with pytest.raises(ValueError):
        session.verify()


def test_source_session_verifies_once_and_returns_defensive_inputs(
    qualified, monkeypatch
):
    from arblab.hyperliquid_copy.qualified_source_session import QualifiedSourceSession
    from arblab.hyperliquid_copy.qualified_day import QualifiedFile

    original, checked = QualifiedFile.verify, []

    def verify(entry):
        checked.append(entry.path)
        return original(entry)

    monkeypatch.setattr(QualifiedFile, "verify", verify)
    session = QualifiedSourceSession(qualified)
    assert session.origin.isoformat() == "2026-08-01T00:00:00+00:00"
    assert session.finish.isoformat() == "2026-08-04T00:00:00+00:00"
    assert session.coins == ("BTC",)
    source_files = json.loads(Path(qualified["path"]).read_text())["files"]
    assert set(checked) == {Path(f["path"]) for f in source_files}
    before = len(checked)
    expected = session.inputs()
    changed = session.inputs()
    changed["pin"]["sha256"] = "0" * 64
    changed["coins"].append("ETH")
    session.verify()
    session.verify()
    assert session.inputs() == expected
    assert len(checked) == before
    assert QualifiedSourceSession(qualified).inputs() == expected
    assert len(checked) > before


@pytest.mark.parametrize(
    "fault", ["source", "report", "caller", "engine", "missing", "symlink", "hardlink"]
)
def test_source_session_rejects_changes(qualified, tmp_path, monkeypatch, fault):
    from arblab.hyperliquid_copy import qualified_source_session as module

    session = module.QualifiedSourceSession(qualified)
    path = first_file(qualified)
    if fault in ("source", "report"):
        target = path if fault == "source" else Path(qualified["path"])
        with target.open("ab") as handle:
            handle.write(b"changed")
    elif fault == "caller":
        qualified["sha256"] = "0" * 64
    elif fault == "engine":
        monkeypatch.setattr(module, "_engine", lambda: "changed")
    elif fault == "missing":
        path.rename(tmp_path / "moved.parquet")
    elif fault == "symlink":
        moved = tmp_path / "moved.parquet"
        path.rename(moved)
        path.symlink_to(moved)
    else:
        (tmp_path / "hardlink.parquet").hardlink_to(path)
    with pytest.raises(ValueError):
        session.verify()


def test_source_session_rejects_initial_corruption(qualified):
    from arblab.hyperliquid_copy.qualified_source_session import QualifiedSourceSession

    with first_file(qualified).open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError):
        QualifiedSourceSession(qualified)


@pytest.mark.parametrize("fault", ["source", "caller", "engine"])
def test_source_session_rechecks_after_expensive_initial_verification(
    qualified, monkeypatch, fault
):
    from arblab.hyperliquid_copy import qualified_source_session as module
    from arblab.hyperliquid_copy.qualified_day import QualifiedFile

    original = QualifiedFile.verify

    def verify(entry):
        original(entry)
        if fault == "source":
            with entry.path.open("ab") as handle:
                handle.write(b"late mutation")
        elif fault == "caller":
            qualified["sha256"] = "0" * 64
        else:
            monkeypatch.setattr(module, "_engine", lambda: "changed")

    monkeypatch.setattr(QualifiedFile, "verify", verify)
    with pytest.raises(ValueError):
        module.QualifiedSourceSession(qualified)
