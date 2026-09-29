import pytest
from types import SimpleNamespace

from arblab.hyperliquid_copy.archive import archive_keys
from arblab.hyperliquid_copy.download import BUCKET
from arblab.hyperliquid_copy.proxy_archive_download import download_archive
from .test_proxy_archive_download import Source as PlainSource


class Source(PlainSource):
    meta = SimpleNamespace(config=SimpleNamespace(retries={"total_max_attempts": 1}))


def objects():
    return [
        dict(key=k, bytes=3, etag='"fixed"')
        for d in ("2026-08-01", "2026-08-02")
        for k in archive_keys(d)
    ]


def test_lifetime_budget_spans_batches_and_restart(tmp_path):
    from arblab.hyperliquid_copy.archive_budget import (
        ArchiveBudget,
        BudgetedArchiveSource,
    )

    path = tmp_path / "budget.sqlite3"
    source = Source()
    budget = ArchiveBudget(path, objects(), 144)
    download_archive(
        BudgetedArchiveSource(source, budget),
        "2026-08-01",
        "2026-08-02",
        tmp_path / "raw",
        max_bytes=72,
    )
    assert budget.reserved_bytes == 72
    reopened = ArchiveBudget(path, objects(), 144)
    download_archive(
        BudgetedArchiveSource(source, reopened),
        "2026-08-02",
        "2026-08-03",
        tmp_path / "raw",
        max_bytes=72,
    )
    assert reopened.reserved_bytes == 144
    assert len(source.gets) == 48
    with pytest.raises(ValueError, match="reserved"):
        reopened.reserve(objects()[0]["key"])
    assert reopened.reserved_bytes == 144


def test_budget_exhaustion_and_failed_request_never_reset_spending(tmp_path):
    from arblab.hyperliquid_copy.archive_budget import (
        ArchiveBudget,
        BudgetedArchiveSource,
    )

    class Broken(Source):
        def get_object(self, **kwargs):
            self.gets.append(kwargs["Key"])
            raise RuntimeError("Network failed")

    source = Broken()
    budget = ArchiveBudget(tmp_path / "budget.sqlite3", objects(), 3)
    adapter = BudgetedArchiveSource(source, budget)

    def get(key):
        return adapter.get_object(
            Bucket=BUCKET, Key=key, RequestPayer="requester", IfMatch='"fixed"'
        )

    with pytest.raises(RuntimeError):
        get(objects()[0]["key"])
    assert budget.reserved_bytes == 3
    with pytest.raises(ValueError, match="reserved"):
        get(objects()[0]["key"])
    with pytest.raises(ValueError, match="budget"):
        get(objects()[1]["key"])
    assert len(source.gets) == 1


@pytest.mark.parametrize("change", ["scope", "budget"])
def test_existing_job_scope_and_limit_cannot_change(tmp_path, change):
    from arblab.hyperliquid_copy.archive_budget import ArchiveBudget

    path = tmp_path / "budget.sqlite3"
    ArchiveBudget(path, objects(), 144)
    with pytest.raises(ValueError, match="changed"):
        ArchiveBudget(
            path,
            objects()[:-1] if change == "scope" else objects(),
            145 if change == "budget" else 144,
        )


def test_adapter_rejects_identity_changes_and_unscoped_requests(tmp_path):
    from arblab.hyperliquid_copy.archive_budget import (
        ArchiveBudget,
        BudgetedArchiveSource,
    )

    source = Source(declared=4)
    budget = ArchiveBudget(tmp_path / "budget.sqlite3", objects(), 144)
    adapter = BudgetedArchiveSource(source, budget)
    args = dict(Bucket=BUCKET, Key=objects()[0]["key"], RequestPayer="requester")
    with pytest.raises(ValueError, match="identity"):
        adapter.head_object(**args)
    for change in (
        {"Key": "outside"},
        {"Bucket": "other"},
        {"IfMatch": "changed"},
        {"Range": "bytes=0-1"},
    ):
        with pytest.raises(ValueError):
            adapter.get_object(**(args | {"IfMatch": '"fixed"'} | change))
    assert budget.reserved_bytes == 0
    assert not source.gets


def test_concurrent_reservations_charge_one_request_only(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from arblab.hyperliquid_copy.archive_budget import ArchiveBudget

    path = tmp_path / "budget.sqlite3"
    first = ArchiveBudget(path, objects(), 144)
    second = ArchiveBudget(path, objects(), 144)

    def reserve(budget):
        try:
            budget.reserve(objects()[0]["key"])
            return "reserved"
        except ValueError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(reserve, [first, second])) == [
            "rejected",
            "reserved",
        ]
    assert first.reserved_bytes == second.reserved_bytes == 3


def test_retrying_client_is_refused_before_network(tmp_path):
    from arblab.hyperliquid_copy.archive_budget import (
        ArchiveBudget,
        BudgetedArchiveSource,
    )

    source = Source()
    source.meta = SimpleNamespace(
        config=SimpleNamespace(retries={"total_max_attempts": 2})
    )
    with pytest.raises(ValueError, match="retries"):
        BudgetedArchiveSource(
            source, ArchiveBudget(tmp_path / "budget.sqlite3", objects(), 144)
        )
    assert not source.gets


def test_batch_resume_does_not_reserve_completed_objects_twice(tmp_path):
    from arblab.hyperliquid_copy.archive_budget import (
        ArchiveBudget,
        BudgetedArchiveSource,
    )
    from arblab.hyperliquid_copy.proxy_archive_download import resume_archive

    path = tmp_path / "budget.sqlite3"
    source = Source()
    budget = ArchiveBudget(path, objects(), 144)

    def stop(item):
        if item.get("downloaded") == 1:
            raise RuntimeError("Pause")

    with pytest.raises(RuntimeError):
        download_archive(
            BudgetedArchiveSource(source, budget),
            "2026-08-01",
            "2026-08-02",
            tmp_path / "raw",
            max_bytes=72,
            progress=stop,
        )
    manifest = next((tmp_path / "raw").glob("*/manifest.json"))
    reopened = ArchiveBudget(path, objects(), 144)
    resume_archive(BudgetedArchiveSource(source, reopened), manifest)
    assert reopened.reserved_bytes == 72
    assert len(source.gets) == 24


@pytest.mark.parametrize("limit", [True, 0, -1, 513 * 1024**3])
def test_invalid_budget_does_not_create_database(tmp_path, limit):
    from arblab.hyperliquid_copy.archive_budget import ArchiveBudget

    path = tmp_path / "budget.sqlite3"
    with pytest.raises(ValueError):
        ArchiveBudget(path, objects(), limit)
    assert not path.exists()
