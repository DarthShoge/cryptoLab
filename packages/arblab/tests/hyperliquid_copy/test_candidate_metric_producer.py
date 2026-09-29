from dataclasses import replace
from datetime import timedelta
from itertools import chain

import pytest

from arblab.hyperliquid_copy.lab_config import METRICS
from arblab.hyperliquid_copy.lab_ranking import rank_activity
from .test_lab_ranking import settings
from .test_streaming_wallet_metrics import mixed_history
from .test_candidate_day import resources
from .test_qualified_day import qualified


def build(resources, qualified, **kwargs):
    from datetime import datetime, timezone
    from arblab.hyperliquid_copy.candidate_metric_producer import (
        build_candidate_metrics,
    )

    config = kwargs.pop(
        "config",
        settings(lookback_days=1, min_volume=0, metric_weights={"gross_volume": 1}),
    )
    decision = kwargs.pop("decision", datetime(2026, 8, 3, tzinfo=timezone.utc))
    return build_candidate_metrics(
        resources,
        qualified,
        "2026-08-01",
        decision,
        config,
        "BTC",
        "gross_excludes_fee",
        **kwargs,
    )


def metric_publications(resources):
    import json

    with resources._connect() as db:
        return [
            d
            for (raw,) in db.execute("SELECT descriptor FROM publications")
            if (d := json.loads(raw))["kind"] == "candidate_metrics"
        ]


def test_budgeted_builder_publishes_all_metrics_and_reuses(
    qualified, resources, tmp_path, monkeypatch
):
    import pyarrow.parquet as pq
    from arblab.hyperliquid_copy import candidate_metric_producer as module
    from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts

    inputs = build(resources, qualified)
    publication = PublishedArtifacts(resources).lookup("candidate_metrics", inputs)
    actual = pq.read_table(resources.root / publication.artifacts[0].path).to_pylist()
    assert len(actual) == 2
    assert all(r["metrics"]["gross_volume"] > 0 for r in actual)
    before = resources.audit()
    assert before["reserved_bytes"] == 0
    assert not list((resources.root / "scratch").iterdir())

    def unexpected(*args, **kwargs):
        raise AssertionError("metric cache hit repeated wallet computation")

    monkeypatch.setattr(module, "merge_metric_rows", unexpected)
    assert build(resources, qualified) == inputs
    assert resources.audit() == before


def test_all_dormant_universe_is_preserved(qualified, resources):
    from datetime import datetime, timezone
    import pyarrow.parquet as pq
    from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts

    inputs = build(
        resources, qualified, decision=datetime(2026, 8, 4, tzinfo=timezone.utc)
    )
    publication = PublishedArtifacts(resources).lookup("candidate_metrics", inputs)
    rows = pq.read_table(resources.root / publication.artifacts[0].path).to_pylist()
    assert len(rows) == 2
    assert all(
        r["metrics"]["gross_volume"] == 0
        and r["exclusions"] == ["no_activity_in_lookback"]
        for r in rows
    )
    assert resources.audit()["reserved_bytes"] == 0


def test_metric_footer_exhaustion_does_not_publish(qualified, resources):
    with pytest.raises(ValueError, match="byte limit"):
        build(resources, qualified, max_bytes=10)
    assert not metric_publications(resources)
    assert resources.audit()["reserved_bytes"] >= 10


def test_working_set_reserved_before_first_sort(qualified, resources, monkeypatch):
    from arblab.hyperliquid_copy import candidate_metric_producer as module

    original = module.OrderedWalletPartitions.plan
    observed = []

    def inspect(reader, **kwargs):
        observed.append(resources.audit()["reserved_bytes"])
        return original(reader, **kwargs)

    monkeypatch.setattr(module.OrderedWalletPartitions, "plan", inspect)
    build(resources, qualified)
    assert observed and all(value >= 3 * 1024**3 + 512 * 1024**2 for value in observed)


def test_builder_amortizes_source_hashing_across_partitions(
    qualified, resources, monkeypatch
):
    from arblab.hyperliquid_copy.qualified_window import QualifiedWindow

    original = QualifiedWindow.verify
    calls = []

    def counted(source):
        calls.append(source)
        return original(source)

    monkeypatch.setattr(QualifiedWindow, "verify", counted)
    build(resources, qualified, max_partition_rows=4)
    assert len(calls) == 4  # Constructor, replay entry/exit, final publication check.


def test_source_change_after_metric_write_blocks_publication(
    qualified, resources, monkeypatch
):
    import json
    from pathlib import Path
    from arblab.hyperliquid_copy import candidate_metric_producer as module

    original = module.write_metric_rows

    def changed(*args, **kwargs):
        token = original(*args, **kwargs)
        path = Path(json.loads(Path(qualified["path"]).read_text())["files"][0]["path"])
        with path.open("ab") as stream:
            stream.write(b"changed")
        return token

    monkeypatch.setattr(module, "write_metric_rows", changed)
    with pytest.raises(ValueError):
        build(resources, qualified)
    assert not metric_publications(resources)


def test_scratch_replacement_after_write_cannot_publish(
    qualified, resources, monkeypatch
):
    from arblab.hyperliquid_copy import candidate_metric_producer as module

    original = module.write_metric_rows

    def changed(*args, **kwargs):
        token = original(*args, **kwargs)
        scratch = next((resources.root / "scratch").iterdir())
        spill = scratch / "spill"
        spill.rename(scratch / "displaced-spill")
        spill.mkdir()
        return token

    monkeypatch.setattr(module, "write_metric_rows", changed)
    with pytest.raises(ValueError, match="spill identity"):
        build(resources, qualified)
    assert not metric_publications(resources)
    assert resources.audit()["reserved_bytes"] >= 3 * 1024**3


def test_interrupted_merge_retains_owned_payload_and_scratch_charge(
    qualified, resources, monkeypatch
):
    from arblab.hyperliquid_copy import candidate_metric_producer as module

    original = module.merge_metric_rows

    def interrupted(*args, **kwargs):
        with __import__("contextlib").closing(original(*args, **kwargs)) as rows:
            yield next(rows)
            raise RuntimeError("simulated metric interruption")

    monkeypatch.setattr(module, "merge_metric_rows", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        build(resources, qualified)
    assert not metric_publications(resources)
    assert resources.audit()["reserved_bytes"] >= 3 * 1024**3 + 512 * 1024**2
    assert list((resources.root / "scratch").glob("*/*.parquet"))


def test_exhausted_shared_budget_prevents_sort(qualified, resources, monkeypatch):
    from datetime import datetime, timezone
    from arblab.hyperliquid_copy.candidate_history import build_candidate_history
    from arblab.hyperliquid_copy import candidate_metric_producer as module

    build_candidate_history(
        resources,
        qualified,
        "2026-08-01",
        datetime(2026, 8, 3, tzinfo=timezone.utc),
        ["BTC"],
        "BTC",
    )
    resources.reserve("staging/" + "e" * 32, 6 * 1024**3, "payload")

    def unexpected(*args, **kwargs):
        raise AssertionError("unbudgeted sort")

    monkeypatch.setattr(module.OrderedWalletPartitions, "plan", unexpected)
    with pytest.raises(ValueError, match="budget"):
        build(resources, qualified)
    assert not metric_publications(resources)


def test_build_and_score_matches_reference_and_weekly_reuses(
    qualified, resources, tmp_path
):
    import json
    from pathlib import Path
    from datetime import datetime, timezone
    from arblab.hyperliquid_copy.candidate_metric_producer import (
        build_and_score_candidates,
    )
    from arblab.hyperliquid_copy.proxy_activity import ProxyActivity, ORDER
    from arblab.hyperliquid_copy.contracts import FillEvent
    from arblab.hyperliquid_copy.proxy_compact import COLUMNS
    from itertools import groupby

    decision = datetime(2026, 8, 3, tzinfo=timezone.utc)
    config = settings(lookback_days=1, min_volume=0, metric_weights={"gross_volume": 1})
    paths = [
        Path(e["path"])
        for e in json.loads(Path(qualified["path"]).read_text())["files"]
    ]
    with ProxyActivity(paths, temp_root=tmp_path) as reference:
        fills = [
            FillEvent(**dict(zip(COLUMNS, row)))
            for row in reference.db.execute(
                f"SELECT {','.join(COLUMNS)} FROM fills WHERE exchange_time>=? AND exchange_time<? ORDER BY user,{ORDER}",
                [decision - timedelta(days=1), decision],
            ).fetchall()
        ]
    expected = rank_activity(
        [(u, list(g)) for u, g in groupby(fills, key=lambda f: f.user)],
        decision,
        config,
        "BTC",
        "gross_excludes_fee",
    )
    args = (resources, qualified, "2026-08-01", decision)
    result = build_and_score_candidates(*args, config, "BTC", "gross_excludes_fee")
    rows = [row for batch in result.iter_batches() for row in batch]
    assert len(rows) == len(expected) == result.candidate_count
    for actual, reference in zip(rows, expected, strict=True):
        for key in (
            "user",
            "score",
            "rank",
            "selected",
            "eligible",
            "weight",
            "exclusions",
            "percentiles",
        ):
            assert actual[key] == reference[key]
        assert actual["metrics"] == {k: reference["metrics"].get(k) for k in METRICS}
    before = resources.audit()
    assert (
        build_and_score_candidates(
            *args, replace(config, reselection="weekly"), "BTC", "gross_excludes_fee"
        )
        == result
    )
    assert resources.audit() == before


@pytest.mark.parametrize("semantics", ["gross_excludes_fee", "net_includes_fee"])
def test_complete_merge_matches_reference_including_dormant_and_open_wallets(
    tmp_path, semantics
):
    from arblab.hyperliquid_copy.candidate_metric_producer import merge_metric_rows

    users = ["0x" + f"{i:040x}" for i in range(5)]
    base = mixed_history()
    groups = [
        (
            user,
            [
                replace(f, user=user)
                for f in (base if i == 2 else base[:1] if i == 3 else [])
            ],
        )
        for i, user in enumerate(users)
    ]
    decision = base[-1].exchange_time + timedelta(seconds=1)
    config = settings(
        coins=["BTC", "ETH"], asset_weights={"BTC": 0.5, "ETH": 0.5}, min_volume=1
    )
    expected = rank_activity(groups, decision, config, None, semantics)
    actual = list(
        merge_metric_rows(
            iter(users),
            chain.from_iterable(rows for _, rows in groups),
            decision,
            config,
            semantics,
            temp_root=tmp_path,
        )
    )
    expected = sorted(expected, key=lambda r: r["user"])
    assert actual == [
        dict(
            user=r["user"],
            metrics={k: r["metrics"].get(k) for k in METRICS},
            exclusions=r["exclusions"],
        )
        for r in expected
    ]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "fault", ["missing_first", "missing_last", "duplicate", "reversed"]
)
def test_merge_rejects_incomplete_or_unordered_candidates(tmp_path, fault):
    from arblab.hyperliquid_copy.candidate_metric_producer import merge_metric_rows

    users = ["0x" + f"{i:040x}" for i in range(3)]
    base = mixed_history()[0]
    fills = [replace(base, user=user) for user in users]
    candidates = (
        users[1:]
        if fault == "missing_first"
        else users[:-1]
        if fault == "missing_last"
        else [users[0], *users]
        if fault == "duplicate"
        else users[::-1]
    )
    with pytest.raises(ValueError, match="candidate"):
        list(
            merge_metric_rows(
                iter(candidates),
                iter(fills),
                base.exchange_time + timedelta(seconds=1),
                settings(),
                "gross_excludes_fee",
                temp_root=tmp_path,
            )
        )


def test_empty_merge_has_no_rows_or_scratch(tmp_path):
    from arblab.hyperliquid_copy.candidate_metric_producer import merge_metric_rows

    decision = mixed_history()[-1].exchange_time + timedelta(seconds=1)
    assert (
        list(
            merge_metric_rows(
                iter(()),
                iter(()),
                decision,
                settings(),
                "gross_excludes_fee",
                temp_root=tmp_path,
            )
        )
        == []
    )
    assert list(tmp_path.iterdir()) == []
