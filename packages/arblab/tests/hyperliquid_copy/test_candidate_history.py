from datetime import datetime, timezone
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from .test_candidate_day import resources
from .test_qualified_day import qualified


def at(value):
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def expected_users(pin, decision, coins):
    report = json.loads(Path(pin["path"]).read_text())
    return sorted(
        {
            r["user"]
            for f in report["files"]
            for r in pq.read_table(f["path"]).to_pylist()
            if r["exchange_time"] < decision and r["coin"] in coins
        }
    )


def test_complete_history_keeps_dormant_wallets(qualified, resources):
    from arblab.hyperliquid_copy.candidate_history import CandidateHistory

    # No fills at all on Aug3; both previously observed wallets remain candidates.
    decision = at("2026-08-03T12:00:00")
    with CandidateHistory(
        resources, qualified, "2026-08-01", decision, ["BTC"]
    ) as history:
        assert not history.completed
        assert list(history.rows()) == expected_users(qualified, decision, ["BTC"])
        assert history.completed
        assert history.db is None
    assert resources.audit()["reserved_bytes"] == 0


@pytest.mark.parametrize(
    "stamp", ["2026-08-01T00:00:00", "2026-08-01T00:00:01", "2026-08-04T00:00:00"]
)
def test_strict_cutoffs_and_terminal_midnight(qualified, resources, stamp):
    from arblab.hyperliquid_copy.candidate_history import CandidateHistory

    decision = at(stamp)
    with CandidateHistory(
        resources, qualified, "2026-08-01", decision, ["BTC"], scope="BTC"
    ) as history:
        assert list(history.rows()) == expected_users(qualified, decision, ["BTC"])


def test_later_origin_cannot_omit_dormant_history(qualified, resources):
    from arblab.hyperliquid_copy.candidate_history import CandidateHistory

    with pytest.raises(ValueError, match="origin"):
        with CandidateHistory(
            resources, qualified, "2026-08-02", at("2026-08-03"), ["BTC"]
        ):
            pass
    with resources._connect() as db:
        assert db.execute("SELECT count(*) FROM publications").fetchone()[0] == 0


def test_day_bound_precedes_materializing_the_daily_chain(
    qualified, resources, monkeypatch
):
    from arblab.hyperliquid_copy import candidate_history as module

    report = json.loads(Path(qualified["path"]).read_text())
    monkeypatch.setattr(
        module, "_previous", lambda *args: dict(report, source_end="2030-01-01")
    )

    def forbidden_range(*args):
        raise AssertionError("unbounded day chain materialized before guard")

    monkeypatch.setattr(module, "range", forbidden_range, raising=False)
    with pytest.raises(ValueError, match="day limit"):
        module.CandidateHistory(
            resources, qualified, "2026-08-01", at("2029-01-01"), ["BTC"]
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("decision", at("2026-08-03")),
        ("days", ("2026-08-02",)),
        ("coins", ("xyz:GOLD",)),
        ("scope", "xyz:GOLD"),
        ("publications", ()),
    ],
)
def test_mutated_context_cannot_certify_a_partial_universe(
    qualified, resources, field, value
):
    from arblab.hyperliquid_copy.candidate_history import CandidateHistory

    with CandidateHistory(
        resources, qualified, "2026-08-01", at("2026-08-02"), ["BTC"]
    ) as history:
        setattr(history, field, value)
        with pytest.raises(ValueError, match="context|chain"):
            history.inputs()
        with pytest.raises(ValueError, match="context|chain"):
            list(history.rows())
        assert not history.completed


def test_context_mutation_midstream_prevents_completion(qualified, resources):
    from arblab.hyperliquid_copy.candidate_history import CandidateHistory

    with CandidateHistory(
        resources, qualified, "2026-08-01", at("2026-08-02"), ["BTC"]
    ) as history:
        rows = history.rows()
        next(rows)
        history.decision = at("2026-08-03")
        with pytest.raises(ValueError, match="context"):
            list(rows)
        assert not history.completed


def test_early_close_does_not_certify_completeness(qualified, resources):
    from arblab.hyperliquid_copy.candidate_history import CandidateHistory

    with CandidateHistory(
        resources, qualified, "2026-08-01", at("2026-08-03"), ["BTC"]
    ) as history:
        rows = history.rows()
        assert next(rows).startswith("0x")
    assert not history.completed
    assert history.db is None
    rows.close()
    assert resources.audit()["reserved_bytes"] == 0


def test_mutation_during_iteration_rejects_completion(qualified, resources):
    from arblab.hyperliquid_copy.candidate_history import CandidateHistory

    with CandidateHistory(
        resources, qualified, "2026-08-01", at("2026-08-03"), ["BTC"]
    ) as history:
        rows = history.rows()
        next(rows)
        report = Path(qualified["path"])
        report.write_text(report.read_text() + " ")
        with pytest.raises(ValueError):
            list(rows)
        assert not history.completed


def test_published_history_reuses_without_query_or_charge(
    qualified, resources, monkeypatch
):
    from arblab.hyperliquid_copy import candidate_history as module

    args = (resources, qualified, "2026-08-01", at("2026-08-03"), ["BTC"])
    result = module.build_candidate_history(*args)
    assert pq.read_table(resources.root / result.artifacts[0].path).column(
        "user"
    ).to_pylist() == expected_users(qualified, args[3], ["BTC"])
    before = resources.audit()

    def unexpected(*args, **kwargs):
        raise AssertionError("cached candidate history reran sorting")

    monkeypatch.setattr(module.CandidateHistory, "rows", unexpected)
    assert module.build_candidate_history(*args) == result
    assert resources.audit() == before


def test_footer_failure_keeps_charge_and_no_partial_publication(qualified, resources):
    from arblab.hyperliquid_copy.candidate_history import build_candidate_history

    with pytest.raises(ValueError, match="byte limit"):
        build_candidate_history(
            resources, qualified, "2026-08-01", at("2026-08-03"), ["BTC"], max_bytes=10
        )
    assert resources.audit()["reserved_bytes"] == 10
    assert list((resources.root / "scratch").iterdir()) == []
    with resources._connect() as db:
        assert all(
            json.loads(r[0])["kind"] != "candidate_history"
            for r in db.execute("SELECT descriptor FROM publications")
        )


def test_metric_consumer_begins_after_candidate_sql_closes(
    qualified, resources, monkeypatch
):
    from arblab.hyperliquid_copy import candidate_history as module

    instances = []
    original = module.CandidateHistory.__enter__

    def observed(instance):
        instances.append(instance)
        return original(instance)

    monkeypatch.setattr(module.CandidateHistory, "__enter__", observed)
    publication = module.build_candidate_history(
        resources, qualified, "2026-08-01", at("2026-08-03"), ["BTC"]
    )
    assert instances[0].completed
    assert instances[0].db is None
    assert instances[0].reader is None
    assert not instances[0].active
    with pq.ParquetFile(resources.root / publication.artifacts[0].path) as reader:
        assert (
            sum(batch.num_rows for batch in reader.iter_batches(batch_size=4096)) == 2
        )


def test_after_query_failure_releases_only_owned_empty_scratch(
    qualified, resources, monkeypatch
):
    from arblab.hyperliquid_copy import candidate_history as module

    original = module.CandidateHistory._open_query

    def broken(instance):
        original(instance)
        raise RuntimeError("query interrupted")

    monkeypatch.setattr(module.CandidateHistory, "_open_query", broken)
    with module.CandidateHistory(
        resources, qualified, "2026-08-01", at("2026-08-03"), ["BTC"]
    ) as history:
        with pytest.raises(RuntimeError, match="interrupted"):
            list(history.rows())
        assert not history.completed
    assert resources.audit()["reserved_bytes"] == 0
    assert list((resources.root / "scratch").iterdir()) == []


def test_more_than_100000_candidates_survive_all_layers(tmp_path, resources):
    import lz4.frame
    from arblab.hyperliquid_copy.candidate_history import build_candidate_history
    from arblab.hyperliquid_copy.proxy_archive_download import download_archive
    from arblab.hyperliquid_copy.proxy_archive_import import import_archive
    from arblab.hyperliquid_copy.proxy_compact import compact_history
    from .test_archive_job import inputs
    from .test_prefix_qualification import qualify

    _, source, _, _ = inputs(tmp_path, days=1)
    count = 100_001
    for key, body in list(source.bodies.items()):
        hour = int(key.rsplit("/", 1)[1].split(".")[0])
        data = json.loads(lz4.frame.decompress(body))
        fill = data["events"][0][1]
        data["events"] = [
            [f"0x{i:040x}", dict(fill, coin="BTC" if i % 2 == 0 else "xyz:GOLD")]
            for i in range(hour * 4167, min((hour + 1) * 4167, count))
        ]
        source.bodies[key] = lz4.frame.compress(json.dumps(data).encode() + b"\n")
    raw = download_archive(source, "2026-08-01", "2026-08-02", tmp_path / "downloaded")
    normalized = import_archive(
        raw, ["BTC", "xyz:GOLD"], tmp_path / "normalized", retain_boundary_spill=True
    )
    compact = compact_history(
        normalized, tmp_path / "compact", partitioning="source_day"
    )
    report = qualify([compact], tmp_path)
    publication = build_candidate_history(
        resources, report, "2026-08-01", at("2026-08-02"), ["BTC", "xyz:GOLD"]
    )
    with pq.ParquetFile(resources.root / publication.artifacts[0].path) as reader:
        seen = 0
        for batch in reader.iter_batches(batch_size=4096):
            assert batch.num_rows <= 4096
            for user in batch.column(0).to_pylist():
                assert user == f"0x{seen:040x}"
                seen += 1
    assert seen == count
    assert resources.audit()["reserved_bytes"] == 0
