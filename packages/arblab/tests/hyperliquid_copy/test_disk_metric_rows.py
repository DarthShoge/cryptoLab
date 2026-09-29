import pyarrow.parquet as pq
import pytest

from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
from arblab.hyperliquid_copy.derived_cache_resources import CacheResources
from arblab.hyperliquid_copy.lab_config import METRICS


@pytest.fixture
def resources(tmp_path):
    with CacheLease(tmp_path) as lease:
        yield CacheResources.create(lease, "metric-rows-test")


def row(index=0, value=1.0, exclusions=()):
    return dict(
        user=f"0x{index:040x}",
        metrics=dict.fromkeys(METRICS, value),
        exclusions=list(exclusions),
    )


def output(resources, token):
    with resources._connect() as db:
        relative = db.execute(
            "SELECT path FROM allocations WHERE token=?", (token,)
        ).fetchone()[0]
    return resources.root / relative


def test_streams_over_100000_rows_in_bounded_groups(resources):
    from arblab.hyperliquid_copy.disk_metric_rows import write_metric_rows

    token = write_metric_rows(resources, (row(i) for i in range(100001)), METRICS)
    meta = pq.read_metadata(output(resources, token))
    assert meta.num_rows == 100001
    assert all(meta.row_group(i).num_rows <= 4096 for i in range(meta.num_row_groups))
    assert resources.audit()["reserved_bytes"] == 0


def test_empty_is_typed_and_excluded_null_is_valid(resources):
    from arblab.hyperliquid_copy.disk_metric_rows import (
        write_metric_rows,
        METRIC_ROWS_SCHEMA,
    )

    empty = write_metric_rows(resources, iter(()), METRICS)
    assert pq.read_schema(output(resources, empty)) == METRIC_ROWS_SCHEMA
    token = write_metric_rows(
        resources,
        iter([row(value=None, exclusions=["no_activity_in_lookback"])]),
        METRICS,
    )
    assert pq.read_table(output(resources, token)).to_pylist()[0] == row(
        value=None, exclusions=["no_activity_in_lookback"]
    )


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), True, "1", 2**53 + 1, 10**400]
)
def test_rejects_nonfinite_or_lossy_metrics(resources, value):
    from arblab.hyperliquid_copy.disk_metric_rows import write_metric_rows

    with pytest.raises(ValueError, match="metric"):
        write_metric_rows(resources, iter([row(value=value)]), METRICS)


def test_exact_large_integer_is_preserved(resources):
    from arblab.hyperliquid_copy.disk_metric_rows import write_metric_rows

    token = write_metric_rows(resources, iter([row(value=2**53)]), METRICS)
    assert (
        pq.read_table(output(resources, token)).to_pylist()[0]["metrics"][
            "gross_volume"
        ]
        == 2**53
    )


@pytest.mark.parametrize(
    "change",
    [
        {"user": "bad"},
        {"user": "0x" + "A" * 40},
        {"metrics": {}},
        {"unknown": True},
        {"exclusions": ["x" * 257]},
        {"exclusions": ["x"] * 33},
        {"exclusions": "no_activity"},
    ],
)
def test_rejects_invalid_record(resources, change):
    from arblab.hyperliquid_copy.disk_metric_rows import write_metric_rows

    with pytest.raises(ValueError):
        write_metric_rows(resources, iter([row() | change]), METRICS)


def test_missing_required_metric_is_not_eligible(resources):
    from arblab.hyperliquid_copy.disk_metric_rows import write_metric_rows

    with pytest.raises(ValueError, match="eligible"):
        write_metric_rows(resources, iter([row(value=None)]), METRICS)


@pytest.mark.parametrize(
    "limits,reason", [({"max_rows": 1}, "row limit"), ({"max_bytes": 10}, "byte limit")]
)
def test_failure_stays_reserved_and_unpublished(resources, limits, reason):
    from arblab.hyperliquid_copy.disk_metric_rows import write_metric_rows

    with pytest.raises(ValueError, match=reason):
        write_metric_rows(resources, iter([row(1), row(2)]), METRICS, **limits)
    assert resources.audit()["reserved_bytes"] == limits.get("max_bytes", 512 * 1024**2)
    with resources._connect() as db:
        assert db.execute("SELECT count(*) FROM publications").fetchone()[0] == 0


def test_invalid_bounds_do_not_reserve(resources):
    from arblab.hyperliquid_copy.disk_metric_rows import write_metric_rows

    for kwargs in ({"max_rows": 0}, {"max_bytes": 0}, {"max_rows": True}):
        with pytest.raises(ValueError):
            write_metric_rows(resources, iter(()), METRICS, **kwargs)
    with pytest.raises(ValueError):
        write_metric_rows(resources, iter(()), ("unknown",))
    assert resources.audit()["reserved_bytes"] == 0
