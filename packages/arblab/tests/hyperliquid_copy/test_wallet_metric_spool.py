from statistics import median

import pytest


@pytest.mark.parametrize("buffer_rows", [2, 4096])
def test_ordered_totals_and_exact_medians_with_or_without_spill(tmp_path, buffer_rows):
    from arblab.hyperliquid_copy.wallet_metric_spool import MetricSpool

    values = [1e16, 1.0, -1e16, 4.0, -2.0]
    with MetricSpool(tmp_path, buffer_rows=buffer_rows) as spool:
        directory = spool.directory
        for i, value in enumerate(values):
            spool.add("fill", value, i + 1, 2 * i)
            spool.add("episode", value, 3 * i, i + 1)
        assert spool.total("fill", "a") == sum(values)
        assert spool.total("fill", "b") == sum(range(1, 6))
        assert spool.total("episode", "a", transform=lambda v: max(v, 0)) == sum(
            max(v, 0) for v in values
        )
        assert spool.median("episode", "b") == median([0, 3, 6, 9, 12])
        assert spool.median("episode", "c") == 3
        assert spool.counts == {"fill": 5, "episode": 5}
        assert spool.peak_buffered_rows <= buffer_rows
        assert (spool.disk_bytes > 0) == (buffer_rows == 2)
        with pytest.raises(ValueError, match="final|sealed"):
            spool.add("fill", 1, 2, 3)
    assert not directory.exists()


@pytest.mark.parametrize("buffer_rows", [2, 4096])
def test_even_and_empty_medians(tmp_path, buffer_rows):
    from arblab.hyperliquid_copy.wallet_metric_spool import MetricSpool

    with MetricSpool(tmp_path, buffer_rows=buffer_rows) as spool:
        for value in [4, 1, 3, 2]:
            spool.add("episode", 0, value, value)
        assert spool.median("episode", "b") == 2.5
        assert spool.median("fill", "b") is None
        assert spool.total("fill", "a") == 0


def test_more_than_one_default_batch_is_bounded(tmp_path):
    from arblab.hyperliquid_copy.wallet_metric_spool import MetricSpool

    with MetricSpool(tmp_path) as spool:
        for n in range(10_001):
            spool.add("fill", float(n), 1, 1)
        assert spool.total("fill", "a") == sum(float(n) for n in range(10_001))
        assert spool.peak_buffered_rows <= 4096
        assert spool.disk_bytes > 0


def test_exception_and_byte_failure_remove_only_owned_scratch(tmp_path):
    from arblab.hyperliquid_copy.wallet_metric_spool import MetricSpool

    keep = tmp_path / "keep.txt"
    keep.write_text("caller data")
    with pytest.raises(RuntimeError, match="caller failure"):
        with MetricSpool(tmp_path, buffer_rows=1) as spool:
            spool.add("fill", 1, 2, 3)
            raise RuntimeError("caller failure")
    with pytest.raises(ValueError, match="byte|limit"):
        with MetricSpool(tmp_path, buffer_rows=1, max_bytes=10) as spool:
            spool.add("fill", 1, 2, 3)
            spool.total("fill", "a")
    assert list(tmp_path.iterdir()) == [keep]
    assert keep.read_text() == "caller data"


@pytest.mark.parametrize(
    "kwargs", [{"buffer_rows": 0}, {"buffer_rows": 4097}, {"max_bytes": 0}]
)
def test_invalid_resource_limits(tmp_path, kwargs):
    from arblab.hyperliquid_copy.wallet_metric_spool import MetricSpool

    with pytest.raises(ValueError):
        MetricSpool(tmp_path, **kwargs)


def test_invalid_observation_and_query_fields(tmp_path):
    from arblab.hyperliquid_copy.wallet_metric_spool import MetricSpool

    with MetricSpool(tmp_path) as spool:
        for kind, a in [("other", 1), ("fill", None), ("fill", "1")]:
            with pytest.raises(ValueError):
                spool.add(kind, a, 1, 2)
        with pytest.raises(ValueError):
            spool.total("fill", "bad_column")


def test_row_group_metadata_has_an_independent_bound(tmp_path):
    from arblab.hyperliquid_copy.wallet_metric_spool import MetricSpool

    with pytest.raises(ValueError, match="row.group"):
        with MetricSpool(tmp_path, buffer_rows=1, max_row_groups=2) as spool:
            for _ in range(3):
                spool.add("fill", 0, 0, 0)
    assert list(tmp_path.iterdir()) == []


def test_median_requires_spill_space_plus_reserve(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from arblab.hyperliquid_copy import wallet_metric_spool as module

    with pytest.raises(ValueError, match="free space"):
        with module.MetricSpool(tmp_path, buffer_rows=1) as spool:
            spool.add("episode", 1, 2, 3)
            spool.total("episode", "a")
            monkeypatch.setattr(
                module.shutil,
                "disk_usage",
                lambda _: SimpleNamespace(free=128 * 1024**2),
            )
            spool.median("episode", "b")
    assert list(tmp_path.iterdir()) == []
