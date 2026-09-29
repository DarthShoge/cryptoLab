from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from .test_qualified_day import qualified


def at(value):
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def test_window_selects_complete_exact_event_overlap_including_distant_spill(qualified):
    from arblab.hyperliquid_copy.qualified_window import QualifiedWindow

    window = QualifiedWindow(qualified, at("2026-08-01T12:00"), at("2026-08-03"))
    assert len(window.entries) == 3
    assert sum(f.rows for f in window.entries) == 144
    window.verify()
    assert window.inputs()["start"] == "2026-08-01T12:00:00+00:00"
    assert window.inputs()["end"] == "2026-08-03T00:00:00+00:00"
    assert "/tmp/" not in json.dumps(window.inputs())


def test_empty_window_retains_verified_witness(qualified):
    from arblab.hyperliquid_copy.qualified_window import QualifiedWindow

    window = QualifiedWindow(qualified, at("2026-08-03"), at("2026-08-04"))
    assert window.entries == ()
    assert window.witness.rows > 0
    window.verify()


def test_upper_boundary_pruning_is_exclusive(qualified):
    from arblab.hyperliquid_copy.qualified_window import QualifiedWindow

    first = QualifiedWindow(qualified, at("2026-08-01"), at("2026-08-02"))
    second = QualifiedWindow(qualified, at("2026-08-02"), at("2026-08-03"))
    assert len(first.entries) == 2
    assert len(second.entries) == 1
    assert not {e.path for e in first.entries} & {e.path for e in second.entries}


def test_changed_derivation_engine_rejected(qualified, monkeypatch):
    from arblab.hyperliquid_copy import qualified_window as module

    window = module.QualifiedWindow(qualified, at("2026-08-01"), at("2026-08-03"))
    monkeypatch.setattr(module, "_code", lambda: (("changed", "0" * 64),))
    with pytest.raises(ValueError, match="engine"):
        window.verify()


def test_qualification_engine_change_during_file_verification_rejected(
    qualified, monkeypatch
):
    from arblab.hyperliquid_copy import qualified_day as source_module
    from arblab.hyperliquid_copy import qualified_window as window_module
    from arblab.hyperliquid_copy.qualified_window import QualifiedWindow

    window = QualifiedWindow(qualified, at("2026-08-01"), at("2026-08-03"))
    original = source_module.QualifiedFile.verify
    engine = source_module._engine()

    def changed(entry):
        original(entry)
        monkeypatch.setattr(source_module, "_engine", lambda: dict(engine, version=999))
        monkeypatch.setattr(window_module, "_engine", lambda: dict(engine, version=999))

    monkeypatch.setattr(source_module.QualifiedFile, "verify", changed)
    with pytest.raises(ValueError, match="engine"):
        window.verify()


@pytest.mark.parametrize(
    "start,end",
    [
        ("2026-07-31", "2026-08-02"),
        ("2026-08-03", "2026-08-05"),
        ("2026-08-02", "2026-08-02"),
        ("2026-08-03", "2026-08-02"),
    ],
)
def test_invalid_window_rejected(qualified, start, end):
    from arblab.hyperliquid_copy.qualified_window import QualifiedWindow

    with pytest.raises(ValueError):
        QualifiedWindow(qualified, at(start), at(end))


def test_context_is_immutable(qualified):
    from arblab.hyperliquid_copy.qualified_window import QualifiedWindow

    window = QualifiedWindow(qualified, at("2026-08-01"), at("2026-08-03"))
    with pytest.raises(FrozenInstanceError):
        window.end = at("2026-08-04")
    first = window.inputs()
    first["coins"].clear()
    assert window.inputs()["coins"] == ["BTC"]


@pytest.mark.parametrize("fault", ["source", "report", "pin", "symlink"])
def test_source_identity_rechecked(qualified, tmp_path, fault):
    from arblab.hyperliquid_copy.qualified_window import QualifiedWindow

    if fault == "pin":
        with pytest.raises(ValueError):
            QualifiedWindow(
                dict(qualified, sha256="0" * 64), at("2026-08-01"), at("2026-08-03")
            )
        return
    window = QualifiedWindow(qualified, at("2026-08-01"), at("2026-08-03"))
    path = Path(qualified["path"]) if fault == "report" else window.entries[0].path
    if fault == "symlink":
        moved = tmp_path / "moved.parquet"
        path.rename(moved)
        path.symlink_to(moved)
    else:
        with path.open("ab") as stream:
            stream.write(b"changed")
    with pytest.raises(ValueError):
        window.verify()
