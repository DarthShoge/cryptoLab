from dataclasses import fields
from datetime import datetime, timezone

import pytest

from arblab.hyperliquid_copy.contracts import FillEvent
from arblab.hyperliquid_copy.proxy_activity import ProxyActivity, ORDER
from arblab.hyperliquid_copy.qualified_window import QualifiedWindow
from .test_qualified_day import qualified


def window(pin):
    return QualifiedWindow(
        pin,
        datetime(2026, 8, 1, tzinfo=timezone.utc),
        datetime(2026, 8, 3, tzinfo=timezone.utc),
    )


def test_source_view_keeps_lazy_parquet_scan_not_materialized_window(
    qualified, tmp_path
):
    from arblab.hyperliquid_copy.ordered_wallet_partitions import (
        OrderedWalletPartitions,
    )

    scratch = tmp_path / "ordered"
    scratch.mkdir()
    reader = OrderedWalletPartitions(window(qualified), ["BTC"], scratch)
    with reader._query() as db:
        explanation = db.execute(
            "EXPLAIN SELECT count(*), min(user), max(user) FROM source"
        ).fetchone()[1]
    assert "COLUMN_DATA_SCAN" not in explanation
    assert "PARQUET" in explanation


def test_verified_batch_hashes_source_before_and_after_not_per_partition(
    qualified, tmp_path, monkeypatch
):
    from arblab.hyperliquid_copy.ordered_wallet_partitions import (
        OrderedWalletPartitions,
    )

    scratch = tmp_path / "ordered"
    scratch.mkdir()
    reader = OrderedWalletPartitions(window(qualified), ["BTC"], scratch)
    calls = []
    original = QualifiedWindow.verify

    def counted(source):
        calls.append(source)
        return original(source)

    monkeypatch.setattr(QualifiedWindow, "verify", counted)
    with reader.verified_batch():
        for index, part in enumerate(reader.plan(max_rows=4)):
            if part.physical_rows:
                artifact = reader.write(part, scratch / f"part-{index}.parquet")
                assert list(reader.read(artifact))
    assert calls == [reader.window, reader.window]


def test_verified_batch_rechecks_source_even_on_interruption(qualified, tmp_path):
    from arblab.hyperliquid_copy.ordered_wallet_partitions import (
        OrderedWalletPartitions,
    )

    scratch = tmp_path / "ordered"
    scratch.mkdir()
    reader = OrderedWalletPartitions(window(qualified), ["BTC"], scratch)
    with pytest.raises(ValueError):
        with reader.verified_batch():
            reader.plan()
            with reader.window.entries[0].path.open("ab") as stream:
                stream.write(b"changed")
            raise RuntimeError("interrupted")


def test_partitions_cover_domain_and_do_not_split_whale_wallet(qualified, tmp_path):
    from arblab.hyperliquid_copy.ordered_wallet_partitions import (
        OrderedWalletPartitions,
    )

    scratch = tmp_path / "ordered"
    scratch.mkdir()
    reader = OrderedWalletPartitions(window(qualified), ["BTC"], scratch)
    plan = reader.plan(max_rows=4)
    assert plan[0].lower == "0x" + "0" * 40
    assert plan[-1].upper == "0y"
    assert all(a.upper == b.lower for a, b in zip(plan, plan[1:]))
    assert sum(p.physical_rows for p in plan) == 144
    whales = [p for p in plan if p.physical_rows > 4]
    assert len(whales) == 2
    assert all(p.single_wallet is not None for p in whales)
    assert reader.db is None


def test_complete_native_order_and_dedup_match_reference(qualified, tmp_path):
    from arblab.hyperliquid_copy.ordered_wallet_partitions import (
        OrderedWalletPartitions,
    )

    source = window(qualified)
    scratch = tmp_path / "ordered"
    scratch.mkdir()
    reader = OrderedWalletPartitions(source, ["BTC"], scratch)
    columns = [f.name for f in fields(FillEvent) if f.name != "raw_details_json"]
    with ProxyActivity(
        [f.path for f in source.entries], temp_root=tmp_path
    ) as reference:
        expected = reference.db.execute(
            f"SELECT {','.join(columns)} FROM fills ORDER BY user,{ORDER}"
        ).fetchall()
    actual = []
    for index, part in enumerate(reader.plan(max_rows=4)):
        if part.physical_rows:
            artifact = reader.write(part, scratch / f"part-{index}.parquet")
            assert reader.db is None
            actual.extend(
                tuple(getattr(f, name) for name in columns)
                for f in reader.read(artifact)
            )
    assert actual == expected
    assert len(actual) == 96


def test_capped_sort_does_not_truncate(qualified, tmp_path):
    from arblab.hyperliquid_copy.ordered_wallet_partitions import (
        OrderedWalletPartitions,
    )

    scratch = tmp_path / "ordered"
    scratch.mkdir()
    reader = OrderedWalletPartitions(window(qualified), ["BTC"], scratch)
    part = reader.plan()[0]
    with pytest.raises(ValueError, match="byte limit"):
        reader.write(part, scratch / "too-small.parquet", max_bytes=10)
    assert reader.db is None


def test_changed_sorted_artifact_rejected(qualified, tmp_path):
    from arblab.hyperliquid_copy.ordered_wallet_partitions import (
        OrderedWalletPartitions,
    )

    scratch = tmp_path / "ordered"
    scratch.mkdir()
    reader = OrderedWalletPartitions(window(qualified), ["BTC"], scratch)
    artifact = reader.write(reader.plan()[0], scratch / "part.parquet")
    with artifact.path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError):
        list(reader.read(artifact))


def test_empty_interval_has_explicit_full_domain_plan(qualified, tmp_path):
    from arblab.hyperliquid_copy.ordered_wallet_partitions import (
        OrderedWalletPartitions,
    )

    source = QualifiedWindow(
        qualified,
        datetime(2026, 8, 3, tzinfo=timezone.utc),
        datetime(2026, 8, 4, tzinfo=timezone.utc),
    )
    scratch = tmp_path / "ordered"
    scratch.mkdir()
    reader = OrderedWalletPartitions(source, ["BTC"], scratch)
    plan = reader.plan()
    assert len(plan) == 1 and plan[0].physical_rows == 0
    artifact = reader.write(plan[0], scratch / "empty.parquet")
    assert list(reader.read(artifact)) == []


@pytest.mark.parametrize("replace_path", [False, True])
def test_reserved_scratch_identity_cannot_change(qualified, tmp_path, replace_path):
    from arblab.hyperliquid_copy.ordered_wallet_partitions import (
        OrderedWalletPartitions,
    )

    scratch = tmp_path / "ordered"
    scratch.mkdir()
    reader = OrderedWalletPartitions(window(qualified), ["BTC"], scratch)
    replacement = tmp_path / "replacement"
    if replace_path:
        scratch.rename(replacement)
        scratch.mkdir()
    else:
        replacement.mkdir()
        reader.scratch = replacement
    with pytest.raises(ValueError, match="scratch"):
        reader.plan()


def test_partition_plan_limit_fails_without_partial_plan(
    qualified, tmp_path, monkeypatch
):
    from arblab.hyperliquid_copy import ordered_wallet_partitions as module

    scratch = tmp_path / "ordered"
    scratch.mkdir()
    reader = module.OrderedWalletPartitions(window(qualified), ["BTC"], scratch)
    monkeypatch.setattr(module, "MAX_PARTS", 1)
    with pytest.raises(ValueError, match="partition"):
        reader.plan(max_rows=4)
    assert reader._plan is None and reader.db is None


@pytest.mark.parametrize("fault", ["source", "engine", "scope"])
def test_changed_inputs_reject_existing_plan(qualified, tmp_path, monkeypatch, fault):
    from arblab.hyperliquid_copy import ordered_wallet_partitions as module

    scratch = tmp_path / "ordered"
    scratch.mkdir()
    reader = module.OrderedWalletPartitions(window(qualified), ["BTC"], scratch)
    part = reader.plan()[0]
    if fault == "source":
        with reader.window.entries[0].path.open("ab") as stream:
            stream.write(b"changed")
    elif fault == "engine":
        monkeypatch.setattr(module.pa, "__version__", "changed")
    else:
        reader.scope = "BTC"
    with pytest.raises(ValueError):
        reader.write(part, scratch / "part.parquet")
    assert not (scratch / "part.parquet").exists()
    assert reader.db is None


def test_row_group_limit_fails_and_closes_sort(qualified, tmp_path, monkeypatch):
    from arblab.hyperliquid_copy import ordered_wallet_partitions as module

    scratch = tmp_path / "ordered"
    scratch.mkdir()
    reader = module.OrderedWalletPartitions(window(qualified), ["BTC"], scratch)
    part = reader.plan()[0]
    monkeypatch.setattr(module, "MAX_GROUPS", 0)
    with pytest.raises(ValueError, match="metadata limit"):
        reader.write(part, scratch / "part.parquet")
    assert reader.db is None


def test_early_reader_close_releases_parquet_handle(qualified, tmp_path, monkeypatch):
    from arblab.hyperliquid_copy import ordered_wallet_partitions as module

    scratch = tmp_path / "ordered"
    scratch.mkdir()
    reader = module.OrderedWalletPartitions(window(qualified), ["BTC"], scratch)
    artifact = reader.write(reader.plan()[0], scratch / "part.parquet")
    opened = []
    original = module.pq.ParquetFile

    def capture(*args, **kwargs):
        handle = original(*args, **kwargs)
        if args[0] == artifact.path:
            opened.append(handle)
        return handle

    monkeypatch.setattr(module.pq, "ParquetFile", capture)
    events = reader.read(artifact)
    next(events)
    events.close()
    assert len(opened) == 1 and opened[0].closed
    assert reader.db is None


@pytest.mark.parametrize("scope", [None, "BTC", "xyz:GOLD"])
def test_cross_market_scope_and_half_open_cutoffs_match_reference(tmp_path, scope):
    import json
    import lz4.frame
    from arblab.hyperliquid_copy.ordered_wallet_partitions import (
        OrderedWalletPartitions,
    )
    from arblab.hyperliquid_copy.proxy_archive_download import download_archive
    from arblab.hyperliquid_copy.proxy_archive_import import import_archive
    from arblab.hyperliquid_copy.proxy_compact import compact_history
    from .test_archive_job import inputs
    from .test_prefix_qualification import qualify

    _, source, _, _ = inputs(tmp_path, days=1)
    for key, body in list(source.bodies.items()):
        hour = int(key.rsplit("/", 1)[1].split(".")[0])
        data = json.loads(lz4.frame.decompress(body))
        for event in data["events"]:
            event[1]["coin"] = "BTC" if hour % 2 else "xyz:GOLD"
        source.bodies[key] = lz4.frame.compress(json.dumps(data).encode() + b"\n")
    raw = download_archive(source, "2026-08-01", "2026-08-02", tmp_path / "downloaded")
    normalized = import_archive(
        raw, ["BTC", "xyz:GOLD"], tmp_path / "normalized", retain_boundary_spill=True
    )
    compact = compact_history(
        normalized, tmp_path / "compact", partitioning="source_day"
    )
    pin = qualify([compact], tmp_path)
    view = QualifiedWindow(
        pin,
        datetime(2026, 8, 1, 4, tzinfo=timezone.utc),
        datetime(2026, 8, 1, 9, tzinfo=timezone.utc),
    )
    scratch = tmp_path / "ordered"
    scratch.mkdir()
    reader = OrderedWalletPartitions(view, ["BTC", "xyz:GOLD"], scratch, scope)
    columns = [f.name for f in fields(FillEvent) if f.name != "raw_details_json"]
    with ProxyActivity([f.path for f in view.entries], temp_root=tmp_path) as reference:
        expected = reference.db.execute(
            f"SELECT {','.join(columns)} FROM fills WHERE exchange_time>=? AND exchange_time<? "
            "AND coin IN (SELECT unnest(?)) " + f"ORDER BY user,{ORDER}",
            [view.start, view.end, [scope] if scope else ["BTC", "xyz:GOLD"]],
        ).fetchall()
    artifact = reader.write(reader.plan()[0], scratch / "part.parquet")
    actual = [
        tuple(getattr(fill, name) for name in columns) for fill in reader.read(artifact)
    ]
    assert actual == expected
    assert len(actual) == (15 if scope is None else 6 if scope == "BTC" else 9)


def test_one_wallet_above_100000_fills_is_streamed_whole(tmp_path):
    import json
    import lz4.frame
    from arblab.hyperliquid_copy.ordered_wallet_partitions import (
        OrderedWalletPartitions,
    )
    from arblab.hyperliquid_copy.proxy_archive_download import download_archive
    from arblab.hyperliquid_copy.proxy_archive_import import import_archive
    from arblab.hyperliquid_copy.proxy_compact import compact_history
    from .test_archive_job import inputs
    from .test_prefix_qualification import qualify

    _, source, _, _ = inputs(tmp_path, days=1)
    count = 100_002
    for key, body in list(source.bodies.items()):
        hour = int(key.rsplit("/", 1)[1].split(".")[0])
        data = json.loads(lz4.frame.decompress(body))
        user, fill = data["events"][0]
        data["events"] = [
            [user, dict(fill, tid=i)]
            for i in range(hour * 4167, min((hour + 1) * 4167, count))
        ]
        source.bodies[key] = lz4.frame.compress(json.dumps(data).encode() + b"\n")
    raw = download_archive(source, "2026-08-01", "2026-08-02", tmp_path / "downloaded")
    normalized = import_archive(
        raw, ["BTC"], tmp_path / "normalized", retain_boundary_spill=True
    )
    compact = compact_history(
        normalized, tmp_path / "compact", partitioning="source_day"
    )
    pin = qualify([compact], tmp_path)
    view = QualifiedWindow(
        pin,
        datetime(2026, 8, 1, tzinfo=timezone.utc),
        datetime(2026, 8, 2, tzinfo=timezone.utc),
    )
    scratch = tmp_path / "ordered"
    scratch.mkdir()
    reader = OrderedWalletPartitions(view, ["BTC"], scratch)
    parts = reader.plan(max_rows=10)
    assert len(parts) == 1 and parts[0].physical_rows == count
    assert parts[0].single_wallet == user
    artifact = reader.write(parts[0], scratch / "whale.parquet")
    seen = 0
    for fill in reader.read(artifact):
        assert fill.tid == seen
        seen += 1
    assert seen == count
    assert reader.db is None
