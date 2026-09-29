from dataclasses import replace
from datetime import timedelta

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from arblab.hyperliquid_copy.lab_config import day
from arblab.hyperliquid_copy.lab_schedule import SelectionState, preview_selection
from arblab.hyperliquid_copy.ranking_artifact import RankingSink, RANKING_SCHEMA
from .test_selection_disk_evidence import disk_result
from .test_lab_pipeline_v2 import dataset


def preview(*args, **kwargs):
    from arblab.hyperliquid_copy.disk_preview import preview_selection_disk

    return preview_selection_disk(*args, **kwargs)


@pytest.mark.parametrize("disk", [False, True])
@pytest.mark.parametrize("scope", [None, "demo:INDEX", "missing"])
@pytest.mark.parametrize("date", ["2026-01-03", "2026-01-04"])
def test_preview_matches_reference_and_preserves_history(tmp_path, disk, scope, date):
    data, config = dataset()
    config = replace(
        config,
        trader=replace(
            config.trader,
            scope="pooled" if scope is None else "per_asset",
            reselection="weekly",
        ),
    )
    at = day(date)
    expected, old, hypothetical = preview_selection(data, config, at, scope)

    class DiskState(SelectionState):
        calls = 0

        def rank_traders(self, at, effective, scope):
            rows = super().rank_traders(at, effective, scope)
            if not rows:
                return rows
            self.calls += 1
            return disk_result(rows, tmp_path / f"source-{self.calls}.parquet")

    artifact, state, actual = preview(
        data,
        config,
        at,
        scope,
        rankings_path=tmp_path / "preview.parquet",
        state_type=DiskState if disk else SelectionState,
    )
    normalized = [
        {k: None if v == {} else v for k, v in row.items()} for row in expected
    ]
    assert pq.read_table(artifact.path).equals(
        pa.Table.from_pylist(normalized, schema=RANKING_SCHEMA)
    )
    assert actual == hypothetical
    assert state.trader_cohorts == old.trader_cohorts
    assert state.market_cohorts == old.market_cohorts
    assert state.market_rankings == old.market_rankings
    assert state.previous == old.previous and state.active == old.active
    assert isinstance(state.rankings, RankingSink)
    artifact.verify()


@pytest.mark.parametrize("date", ["2026-08-10", "2026-08-12"])
def test_scheduled_reader_actual_and_midweek_preview(tmp_path, date):
    from arblab.hyperliquid_copy.lab_pipeline_proxy import ProxySelectionState
    from .test_proxy_weekly import multiweek, weekly_config
    from .test_scheduled_activity import rolling

    data, config = multiweek(tmp_path), weekly_config()
    at = day(date)
    try:
        expected, old, hypothetical = preview_selection(
            data, config, at, "BTC", state_type=ProxySelectionState
        )
    finally:
        data.activity.close()
    data.activity = rolling(tmp_path, config)
    try:
        artifact, state, actual = preview(
            data,
            config,
            at,
            "BTC",
            rankings_path=tmp_path / "scheduled.parquet",
            state_type=ProxySelectionState,
        )
    finally:
        data.activity.close()
    normalized = [
        {k: None if v == {} else v for k, v in row.items()} for row in expected
    ]
    assert pq.read_table(artifact.path).equals(
        pa.Table.from_pylist(normalized, schema=RANKING_SCHEMA)
    )
    assert actual == hypothetical and state.trader_cohorts == old.trader_cohorts


@pytest.mark.parametrize("fault", ["before", "after", "intraday", "naive"])
def test_invalid_decision_never_opens_output(tmp_path, fault):
    data, config = dataset()
    at = day(config.start)
    if fault == "before":
        at -= timedelta(days=1)
    elif fault == "after":
        at = day(config.end)
    elif fault == "intraday":
        at += timedelta(hours=1)
    else:
        at = at.replace(tzinfo=None)
    output = tmp_path / "invalid.parquet"
    with pytest.raises(ValueError):
        preview(data, config, at, None, rankings_path=output, state_type=SelectionState)
    assert not output.exists()


@pytest.mark.parametrize("fault", ["source", "output"])
def test_requested_artifact_and_output_failures_do_not_finalize(
    tmp_path, monkeypatch, fault
):
    data, config = dataset()
    config = replace(config, trader=replace(config.trader, scope="pooled"))
    at = day(config.start)
    sinks = []
    original = RankingSink.__enter__

    def enter(sink):
        sinks.append(sink)
        return original(sink)

    monkeypatch.setattr(RankingSink, "__enter__", enter)

    class DiskState(SelectionState):
        def rank_traders(self, when, effective, scope):
            result = disk_result(
                super().rank_traders(when, effective, scope),
                tmp_path / "source.parquet",
            )
            if fault == "source":
                with result.artifact.path.open("ab") as stream:
                    stream.write(b"changed")
            return result

    if fault == "output":

        def fail(*args):
            raise OSError("output interrupted")

        monkeypatch.setattr(RankingSink, "extend", fail)
    with pytest.raises((ValueError, OSError), match="identity|interrupted"):
        preview(
            data,
            config,
            at,
            None,
            rankings_path=tmp_path / "failed.parquet",
            state_type=DiskState,
        )
    assert len(sinks) == 1
    with pytest.raises(ValueError, match="not successfully finalized"):
        sinks[0].artifact


def test_existing_output_is_not_overwritten(tmp_path):
    data, config = dataset()
    output = tmp_path / "existing.parquet"
    output.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        preview(
            data,
            config,
            day(config.start),
            None,
            rankings_path=output,
            state_type=SelectionState,
        )
    assert output.read_bytes() == b"preserve"


@pytest.mark.parametrize("corrupt", [False, True])
def test_large_disk_preview_keeps_all_exclusions_and_checks_discarded_history(
    tmp_path, monkeypatch, corrupt
):
    data, config = dataset()
    at = day("2026-01-04")
    template_state = SelectionState(data, config)
    template_state.advance(day(config.start))
    template = template_state.rankings[0] | dict(
        eligible=False,
        selected=False,
        rank=None,
        score=None,
        weight=0.0,
        percentiles={},
        reasons=["no_activity_in_lookback"],
        exclusions=["no_activity_in_lookback"],
    )

    class DiskState(SelectionState):
        calls = 0

        def rank_traders(self, when, effective, scope):
            self.calls += 1
            records = [
                template | dict(user=f"0x{i:040x}", decision_time=when, coin=scope)
                for i in range(100001 if when == at else 1)
            ]
            result = disk_result(records, tmp_path / f"large-{self.calls}.parquet")
            if corrupt and when < at:
                with result.artifact.path.open("ab") as stream:
                    stream.write(b"corruption")
            return result

    sizes = []
    original = RankingSink.extend

    def bounded(sink, rows):
        sizes.append(len(rows))
        assert len(rows) <= 4096
        return original(sink, rows)

    monkeypatch.setattr(RankingSink, "extend", bounded)
    if corrupt:
        with pytest.raises(ValueError, match="identity"):
            preview(
                data,
                config,
                at,
                "BTC",
                rankings_path=tmp_path / "output.parquet",
                state_type=DiskState,
            )
    else:
        artifact, _, hypothetical = preview(
            data,
            config,
            at,
            "BTC",
            rankings_path=tmp_path / "output.parquet",
            state_type=DiskState,
        )
        assert not hypothetical and artifact.rows == 100001
        assert sum(sizes) == 100001
        rows = pq.read_table(
            artifact.path, columns=["decision_time", "coin", "exclusions"]
        )
        assert set(rows["decision_time"].to_pylist()) == {at}
        assert set(rows["coin"].to_pylist()) == {"BTC"}
        assert rows["exclusions"][100000].as_py() == ["no_activity_in_lookback"]
