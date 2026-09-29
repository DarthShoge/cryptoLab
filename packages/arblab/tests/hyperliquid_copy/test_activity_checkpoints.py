from dataclasses import asdict, replace
from datetime import timedelta
import json
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from arblab.hyperliquid_copy.compact_catalog import CompactCatalog
from arblab.hyperliquid_copy.download import file_hash
from arblab.hyperliquid_copy.proxy_activity import ProxyActivity
from arblab.hyperliquid_copy.proxy_compact import compact_history
from .test_proxy_compact import source
from .test_proxy_activity import fills
from .test_lab_ranking import settings


def catalog_for(tmp_path, groups):
    catalog = CompactCatalog(tmp_path / "catalog.sqlite3")
    ids, paths = [], []
    for index, rows in enumerate(groups):
        root = tmp_path / str(index)
        root.mkdir()
        manifest, path, _ = source(root)
        pq.write_table(pa.Table.from_pylist([asdict(r) for r in rows]), path)
        meta = json.loads(manifest.read_text())
        meta.update(
            rows=len(rows),
            output_bytes=path.stat().st_size,
            files=[
                dict(
                    name=path.name,
                    rows=len(rows),
                    bytes=path.stat().st_size,
                    sha256=file_hash(path),
                )
            ],
        )
        manifest.write_text(json.dumps(meta))
        compact = compact_history(manifest, root / "compact")
        ids.append(catalog.register(compact))
        paths.append(
            compact.parent / json.loads(compact.read_text())["files"][0]["name"]
        )
    return catalog, ids, paths


def histories():
    old = fills()
    cutoff = max(r.exchange_time for r in old) + timedelta(days=2)
    recent = [
        replace(
            f,
            tid=f.tid + 1000,
            event_id=f.event_id + "recent",
            exchange_time=cutoff + timedelta(hours=i),
        )
        for i, f in enumerate(old[:2])
    ]
    future = replace(
        old[0],
        tid=9999,
        event_id="future",
        user="0x" + "f" * 40,
        exchange_time=cutoff + timedelta(days=5),
    )
    return old, recent + [future], cutoff


def test_restart_and_advance_match_full_history_without_expired_inputs(tmp_path):
    from arblab.hyperliquid_copy.activity_checkpoints import ActivityCheckpoints

    old, recent, cutoff = histories()
    catalog, ids, paths = catalog_for(tmp_path, [old, recent])
    store = ActivityCheckpoints(tmp_path / "checkpoints")
    identity = store.build(catalog, ids, cutoff, temp_root=tmp_path)
    assert store.metadata(identity)["seed"]["rows"] == 6
    with ProxyActivity(paths, temp_root=tmp_path) as full:
        # Immutable seed cache has already been validated; expired files are no
        # longer query inputs. Move, don't delete, to prove they are not read.
        paths[0].rename(paths[0].with_suffix(".retained"))
        store = ActivityCheckpoints(tmp_path / "checkpoints")
        for low in [cutoff, cutoff + timedelta(days=2), cutoff + timedelta(days=6)]:
            if low > cutoff:
                identity = store.advance(identity, low, temp_root=tmp_path)
            end = low + timedelta(days=1)
            with store.open(identity, low, end, temp_root=tmp_path) as cached:
                for r in old + recent:
                    assert cached.position(r.user, "BTC", end) == full.position(
                        r.user, "BTC", end
                    )
                assert cached.observed(end) == full.observed(end)
                assert cached.volume("BTC", low, end) == full.volume("BTC", low, end)
                assert cached.rank(
                    end, settings(lookback_days=1), "BTC", "gross_excludes_fee"
                ) == full.rank(
                    end, settings(lookback_days=1), "BTC", "gross_excludes_fee"
                )
                assert cached.hourly_exposure(
                    old[0].user, "BTC", low, end, max_price_age_seconds=3600
                ) == full.hourly_exposure(
                    old[0].user, "BTC", low, end, max_price_age_seconds=3600
                )


def test_old_conflict_outside_latest_seeds_rejects_build(tmp_path):
    from arblab.hyperliquid_copy.activity_checkpoints import ActivityCheckpoints

    old, recent, cutoff = histories()
    catalog, ids, _ = catalog_for(tmp_path, [old, [replace(old[0], px=999), *recent]])
    store = ActivityCheckpoints(tmp_path / "checkpoints")
    with pytest.raises(ValueError, match="conflicting"):
        store.build(catalog, ids, cutoff, temp_root=tmp_path)
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT count(*) FROM checkpoints").fetchone()[0] == 0


@pytest.mark.parametrize("fault", ["seed", "active", "metadata", "version"])
def test_corruption_rejects_reopen(tmp_path, fault):
    from arblab.hyperliquid_copy.activity_checkpoints import ActivityCheckpoints

    old, recent, cutoff = histories()
    catalog, ids, paths = catalog_for(tmp_path, [old, recent])
    store = ActivityCheckpoints(tmp_path / "checkpoints")
    identity = store.build(catalog, ids, cutoff, temp_root=tmp_path)
    if fault == "seed":
        (store.root / identity / "seeds.parquet").write_bytes(b"corrupt")
    elif fault == "active":
        paths[1].write_bytes(b"corrupt")
    else:
        with sqlite3.connect(store.path) as db:
            if fault == "metadata":
                db.execute("UPDATE checkpoints SET metadata='{}'")
            else:
                db.execute("PRAGMA user_version=999")
        if fault == "version":
            with pytest.raises(ValueError, match="checkpoint"):
                ActivityCheckpoints(store.root)
            return
    with pytest.raises(ValueError, match="identity|metadata"):
        store.open(identity, cutoff, cutoff + timedelta(days=1), temp_root=tmp_path)
    assert not list(tmp_path.glob("proxy_activity_*"))


def test_cutoff_bounds_unknown_ids_and_byte_limits(tmp_path):
    from arblab.hyperliquid_copy.activity_checkpoints import ActivityCheckpoints

    old, recent, cutoff = histories()
    catalog, ids, _ = catalog_for(tmp_path, [old, recent])
    store = ActivityCheckpoints(tmp_path / "checkpoints")
    identity = store.build(catalog, ids, cutoff, temp_root=tmp_path)
    with pytest.raises(ValueError, match="cutoff"):
        store.open(identity, cutoff - timedelta(days=1), cutoff, temp_root=tmp_path)
    with pytest.raises(ValueError, match="cutoff"):
        store.advance(identity, cutoff - timedelta(days=1), temp_root=tmp_path)
    with pytest.raises(ValueError, match="checkpoint"):
        store.open("../bad", cutoff, cutoff + timedelta(days=1), temp_root=tmp_path)
    with pytest.raises(ValueError, match="byte"):
        store.open(
            identity,
            cutoff,
            cutoff + timedelta(days=1),
            temp_root=tmp_path,
            max_input_bytes=1,
        )


def test_failed_publication_leaves_no_reusable_record(tmp_path, monkeypatch):
    from arblab.hyperliquid_copy.activity_checkpoints import ActivityCheckpoints

    old, recent, cutoff = histories()
    catalog, ids, _ = catalog_for(tmp_path, [old, recent])
    store = ActivityCheckpoints(tmp_path / "checkpoints")

    def fail(*args):
        raise OSError("injected publication failure")

    monkeypatch.setattr(store, "_commit", fail)
    with pytest.raises(OSError, match="injected"):
        store.build(catalog, ids, cutoff, temp_root=tmp_path)
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT count(*) FROM checkpoints").fetchone()[0] == 0


def test_overlapping_old_duplicate_cannot_override_seed_order(tmp_path):
    from arblab.hyperliquid_copy.activity_checkpoints import ActivityCheckpoints

    first = replace(fills()[0], source_key="a")
    last = replace(
        first,
        tid=999,
        event_id="last",
        source_key="m",
        post_position=2,
        start_position=1,
    )
    cutoff = first.exchange_time + timedelta(days=1)
    duplicate = replace(first, source_key="z", event_id="duplicate")
    boundary = replace(
        first,
        tid=1000,
        event_id="boundary",
        exchange_time=cutoff,
        start_position=2,
        post_position=3,
    )
    catalog, ids, paths = catalog_for(tmp_path, [[first, last], [duplicate, boundary]])
    store = ActivityCheckpoints(tmp_path / "checkpoints")
    identity = store.build(catalog, ids, cutoff, temp_root=tmp_path)
    paths[0].rename(paths[0].with_suffix(".retained"))
    with store.open(
        identity, cutoff, cutoff + timedelta(days=1), temp_root=tmp_path
    ) as a:
        assert a.position(first.user, "BTC", cutoff) == 2
        assert a.position(first.user, "BTC", cutoff + timedelta(microseconds=1)) == 3
    advanced = store.advance(identity, cutoff + timedelta(days=1), temp_root=tmp_path)
    with store.open(
        advanced,
        cutoff + timedelta(days=1),
        cutoff + timedelta(days=2),
        temp_root=tmp_path,
    ) as a:
        assert a.position(first.user, "BTC", cutoff + timedelta(days=1)) == 3


def test_changed_engine_rejects_cache_and_empty_seed_does_not_invent_positions(
    tmp_path, monkeypatch
):
    import arblab.hyperliquid_copy.activity_checkpoints as module

    old, recent, _ = histories()
    cutoff = min(f.exchange_time for f in old) - timedelta(days=1)
    catalog, ids, _ = catalog_for(tmp_path, [old, recent])
    store = module.ActivityCheckpoints(tmp_path / "checkpoints")
    identity = store.build(catalog, ids, cutoff, temp_root=tmp_path)
    assert store.metadata(identity)["seed"]["rows"] == 0
    with store.open(
        identity, cutoff, cutoff + timedelta(days=1), temp_root=tmp_path
    ) as activity:
        assert activity.observed(cutoff) == []
        assert activity.position(old[0].user, "BTC", cutoff) is None
    monkeypatch.setattr(module, "_engine", lambda: {"schema": 999})
    with pytest.raises(ValueError, match="engine"):
        store.open(identity, cutoff, cutoff + timedelta(days=1), temp_root=tmp_path)


@pytest.mark.parametrize("target", ["directory", "seed", "database"])
def test_symlink_cache_paths_reject(tmp_path, target):
    from arblab.hyperliquid_copy.activity_checkpoints import ActivityCheckpoints

    old, recent, cutoff = histories()
    catalog, ids, _ = catalog_for(tmp_path, [old, recent])
    store = ActivityCheckpoints(tmp_path / "checkpoints")
    identity = store.build(catalog, ids, cutoff, temp_root=tmp_path)
    path = {
        "directory": store.root / identity,
        "seed": store.root / identity / "seeds.parquet",
        "database": store.path,
    }[target]
    moved = path.with_suffix(".retained")
    path.rename(moved)
    path.symlink_to(moved)
    with pytest.raises(ValueError, match="identity|[Ss]ymlink"):
        store.open(identity, cutoff, cutoff + timedelta(days=1), temp_root=tmp_path)


def test_replay_filters_expired_overlap_before_seed_union(tmp_path):
    # Exercise the low-level reader contract independently of cache publication.
    from .test_proxy_activity import partition

    event = fills()[0]
    cutoff = event.exchange_time + timedelta(days=1)
    path = partition(tmp_path, [event])
    with pytest.raises(ValueError, match="together"):
        ProxyActivity([], temp_root=tmp_path, seed_path=path)
    with pytest.raises(ValueError, match="cutoff"):
        ProxyActivity(
            [],
            temp_root=tmp_path,
            seed_path=path,
            seed_cutoff=cutoff,
            query_window=(event.exchange_time, cutoff),
        )
    with pytest.raises(ValueError, match="strictly"):
        ProxyActivity(
            [],
            temp_root=tmp_path,
            seed_path=path,
            seed_cutoff=event.exchange_time,
            query_window=(event.exchange_time, cutoff),
        )


def test_repeated_advance_reuses_frozen_lineage_cutoff(tmp_path):
    from arblab.hyperliquid_copy.activity_checkpoints import ActivityCheckpoints

    old, recent, cutoff = histories()
    catalog, ids, _ = catalog_for(tmp_path, [old, recent])
    store = ActivityCheckpoints(tmp_path / "checkpoints")
    initial = store.build(catalog, ids, cutoff, temp_root=tmp_path)
    next_cutoff = cutoff + timedelta(days=1)
    first = store.advance(initial, next_cutoff, temp_root=tmp_path)
    assert store.advance(initial, next_cutoff, temp_root=tmp_path) == first


@pytest.mark.parametrize("fault", ["budget", "free_space"])
def test_publication_checks_total_cache_and_free_space(tmp_path, monkeypatch, fault):
    import arblab.hyperliquid_copy.activity_checkpoints as module
    from types import SimpleNamespace

    old, recent, cutoff = histories()
    catalog, ids, _ = catalog_for(tmp_path, [old, recent])
    store = module.ActivityCheckpoints(tmp_path / "checkpoints")
    if fault == "budget":
        monkeypatch.setattr(module, "MAX_CACHE_BYTES", 1, raising=False)
    else:
        import shutil

        monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    with pytest.raises(ValueError, match="cache|space"):
        store.build(catalog, ids, cutoff, temp_root=tmp_path)
    assert not list(store.root.glob("*/seeds.parquet"))


@pytest.mark.parametrize("empty", [False, True])
def test_seed_byte_limit_stops_writes_including_parquet_footer(
    tmp_path, monkeypatch, empty
):
    import arblab.hyperliquid_copy.activity_checkpoints as module

    old, recent, cutoff = histories()
    if empty:
        cutoff = min(row.exchange_time for row in old) - timedelta(hours=1)
    catalog, ids, _ = catalog_for(tmp_path, [old, recent])
    store = module.ActivityCheckpoints(tmp_path / "checkpoints")
    monkeypatch.setattr(module, "MAX_SEED_BYTES", 64)
    with pytest.raises(ValueError, match="seed byte"):
        store.build(catalog, ids, cutoff, temp_root=tmp_path)
    assert all(p.stat().st_size <= 64 for p in store.root.glob("*/seeds.parquet"))
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT count(*) FROM checkpoints").fetchone()[0] == 0
