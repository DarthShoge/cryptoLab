from dataclasses import asdict
from datetime import timedelta
from itertools import groupby
import uuid

import pytest

from arblab.hyperliquid_copy.feature_window import FeatureWindow
from .test_candidate_day import resources
from .test_feature_window import days
from .test_feature_day_builder import DAY, SEMANTICS
from .test_qualified_day import qualified


@pytest.fixture
def window(qualified, resources, days):
    return FeatureWindow(
        resources,
        qualified,
        days,
        DAY + timedelta(hours=1),
        DAY + timedelta(days=2, hours=1),
        ["BTC"],
        SEMANTICS,
    )


@pytest.fixture
def scratch(resources):
    relative = "scratch/" + uuid.uuid4().hex
    resources.reserve(relative, 3 * 1024**3, "scratch")
    path = resources.root / relative
    path.mkdir()
    (path / "spill").mkdir()
    return path


def test_complete_ordered_features_match_days_with_query_closed(window, scratch):
    from arblab.hyperliquid_copy.ordered_feature_partitions import (
        OrderedFeaturePartitions,
    )

    reader = OrderedFeaturePartitions(window, scratch)
    expected = [
        row
        for day in window.days
        for row in day.observations()
        if window.start <= row.order_key[0] < window.end
    ]
    expected.sort(
        key=lambda r: (r.user, *r.order_key, 0 if hasattr(r, "fragments") else 1)
    )
    actual = []
    before = window.resources.audit()
    with reader.verified_batch():
        parts = reader.plan(max_rows=4)
        assert sum(p.physical_rows for p in parts) == len(expected)
        for i, part in enumerate(parts):
            if not part.physical_rows:
                continue
            artifact = reader.write(part, scratch / f"part{i}.parquet")
            assert reader.db is None
            for row in reader.read(artifact):
                assert reader.db is None
                actual.append(row)
            artifact.path.unlink()  # Fully consumed test-owned scratch only.
    assert actual == expected
    assert window.resources.audit() == before


def test_ordered_features_feed_exact_metrics(window, scratch):
    from arblab.hyperliquid_copy.ordered_feature_partitions import (
        OrderedFeaturePartitions,
    )
    from arblab.hyperliquid_copy.feature_wallet_metrics import feature_wallet_metrics
    from arblab.hyperliquid_copy.ranking import RankingConfig
    from arblab.hyperliquid_copy.contracts import FillEvent
    from arblab.hyperliquid_copy.proxy_activity import ProxyActivity, ORDER
    from arblab.hyperliquid_copy.proxy_compact import COLUMNS
    from arblab.hyperliquid_copy.streaming_wallet_metrics import stream_wallet_metrics

    reader = OrderedFeaturePartitions(window, scratch)
    config = RankingConfig(lookback_days=2)
    expected = [
        row
        for day in window.days
        for row in day.observations()
        if window.start <= row.order_key[0] < window.end
    ]
    expected.sort(
        key=lambda r: (r.user, *r.order_key, 0 if hasattr(r, "fragments") else 1)
    )
    reference = {}
    for user, rows in groupby(expected, key=lambda r: r.user):
        reference[user] = asdict(
            feature_wallet_metrics(
                rows, window.end, config, SEMANTICS, temp_root=scratch
            )
        )
    actual = {}
    with reader.verified_batch():
        for i, part in enumerate(reader.plan()):
            if not part.physical_rows:
                continue
            artifact = reader.write(part, scratch / f"part{i}.parquet")
            for user, rows in groupby(reader.read(artifact), key=lambda r: r.user):
                assert reader.db is None
                actual[user] = asdict(
                    feature_wallet_metrics(
                        rows, window.end, config, SEMANTICS, temp_root=scratch
                    )
                )
            artifact.path.unlink()  # Fully consumed test-owned scratch only.
    assert actual == reference
    with ProxyActivity(
        [e.path for e in window.source.entries], temp_root=scratch
    ) as source:
        raw = source.db.execute(
            f"SELECT {','.join(COLUMNS)} FROM fills WHERE exchange_time>=? AND exchange_time<? ORDER BY user,{ORDER}",
            [window.start, window.end],
        ).fetchall()
    direct = {}
    for user, rows in groupby(
        (FillEvent(**dict(zip(COLUMNS, r))) for r in raw), key=lambda r: r.user
    ):
        direct[user] = asdict(
            stream_wallet_metrics(
                rows, window.end, config, SEMANTICS, temp_root=scratch
            )
        )
    assert actual == direct


def test_unreserved_scratch_rejected(window, tmp_path):
    from arblab.hyperliquid_copy.ordered_feature_partitions import (
        OrderedFeaturePartitions,
    )

    path = tmp_path / "not_reserved"
    path.mkdir()
    (path / "spill").mkdir()
    with pytest.raises(ValueError):
        OrderedFeaturePartitions(window, path)


def test_partition_bounds_and_single_output_ownership(window, scratch):
    from arblab.hyperliquid_copy.ordered_feature_partitions import (
        OrderedFeaturePartitions,
    )
    from arblab.hyperliquid_copy.ordered_wallet_partitions import AddressPartition

    reader = OrderedFeaturePartitions(window, scratch)
    part = next(p for p in reader.plan() if p.physical_rows)
    with pytest.raises(ValueError, match="bound changed"):
        reader.plan(max_rows=1)
    with pytest.raises(ValueError, match="complete plan"):
        reader.write(AddressPartition("bad", "bad", 0, None), scratch / "bad.parquet")
    with pytest.raises(ValueError):
        reader.write(part, scratch.parent / "outside.parquet")
    artifact = reader.write(part, scratch / "first.parquet")
    with pytest.raises(ValueError, match="One owned"):
        reader.write(part, scratch / "second.parquet")
    assert list(reader.read(artifact))


def test_overflow_retains_scratch_obligation(window, scratch):
    from arblab.hyperliquid_copy.ordered_feature_partitions import (
        OrderedFeaturePartitions,
    )

    reader = OrderedFeaturePartitions(window, scratch)
    part = next(p for p in reader.plan() if p.physical_rows)
    before = window.resources.audit()
    with pytest.raises(ValueError, match="byte limit"):
        reader.write(part, scratch / "partial.parquet", max_bytes=10)
    assert reader.db is None
    assert window.resources.audit() == before
    assert (scratch / "partial.parquet").exists()


@pytest.mark.parametrize("target", ["input", "output"])
def test_early_close_rechecks_mutated_files(window, scratch, target):
    from arblab.hyperliquid_copy.ordered_feature_partitions import (
        OrderedFeaturePartitions,
    )

    reader = OrderedFeaturePartitions(window, scratch)
    part = next(p for p in reader.plan() if p.physical_rows)
    artifact = reader.write(part, scratch / "ordered.parquet")
    rows = reader.read(artifact)
    next(rows)
    path = (
        artifact.path
        if target == "output"
        else window.resources.root / window.observation_pins[0].path
    )
    with path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError):
        rows.close()


def test_window_code_change_during_batch_rejected(window, scratch, monkeypatch):
    from arblab.hyperliquid_copy import ordered_feature_partitions as module

    reader = module.OrderedFeaturePartitions(window, scratch)
    original = module.file_hash
    with reader.verified_batch():
        with monkeypatch.context() as patch:
            patch.setattr(
                module,
                "file_hash",
                lambda path: "0" * 64
                if path.name == "feature_window.py"
                else original(path),
            )
            with pytest.raises(ValueError, match="context"):
                reader.plan()


@pytest.mark.parametrize("target", ["code", "spill"])
def test_mutation_during_final_pinned_check_rejected(
    window, scratch, monkeypatch, target
):
    from arblab.hyperliquid_copy import ordered_feature_partitions as module

    reader = module.OrderedFeaturePartitions(window, scratch)
    original_verify, original_hash = FeatureWindow.verify, module.file_hash
    calls = 0

    def verify(value):
        nonlocal calls
        original_verify(value)
        calls += 1
        if calls == 2:
            if target == "code":
                monkeypatch.setattr(
                    module,
                    "file_hash",
                    lambda path: "0" * 64
                    if path.name == "ordered_feature_partitions.py"
                    else original_hash(path),
                )
            else:
                (scratch / "spill").rename(scratch / "old_spill")
                (scratch / "spill").mkdir()

    monkeypatch.setattr(FeatureWindow, "verify", verify)
    with pytest.raises(ValueError, match="context/directory"):
        with reader._pinned():
            pass
