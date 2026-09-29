from dataclasses import replace
from datetime import timedelta
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from arblab.hyperliquid_copy.disk_score_result import ScoredCohort
from arblab.hyperliquid_copy.lab_config import day
from arblab.hyperliquid_copy.lab_schedule import SelectionState
from arblab.hyperliquid_copy.ranking_artifact import (
    RankingSink,
    RankingFile,
    RANKING_SCHEMA,
)
from arblab.hyperliquid_copy.download import file_hash
from .test_lab_pipeline_v2 import dataset


def disk_result(rows, path, requested=2):
    # Match CohortQuery.write_rankings: {} is a non-null struct with null fields.
    # Run-report RankingSink intentionally converts {} to a null struct instead.
    with pq.ParquetWriter(path, RANKING_SCHEMA, compression="zstd") as writer:
        for offset in range(0, len(rows), 4096):
            writer.write_table(
                pa.Table.from_pylist(
                    rows[offset : offset + 4096], schema=RANKING_SCHEMA
                )
            )
    selected = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
        for r in batch.to_pylist():
            if r["selected"]:
                r["percentiles"] = {
                    k: v for k, v in r["percentiles"].items() if v is not None
                }
                selected.append(
                    json.dumps(
                        {
                            k: v.isoformat() if k == "decision_time" else v
                            for k, v in r.items()
                            if k not in ("market_decision_time", "decision_trigger")
                        }
                    )
                )
    artifact = RankingFile(path, len(rows), path.stat().st_size, file_hash(path))
    return ScoredCohort(
        None,
        artifact,
        len(rows),
        sum(r["eligible"] for r in rows),
        requested,
        tuple(selected),
    )


def clean(rows):
    return [
        r
        | {
            k: {m: v for m, v in r[k].items() if v is not None}
            for k in ("metrics", "percentiles")
        }
        for r in rows
    ]


@pytest.mark.parametrize("scope", ["per_asset", "pooled"])
def test_actual_selection_advance_matches_legacy_rotation(tmp_path, scope):
    data, config = dataset()
    config = replace(
        config, trader=replace(config.trader, scope=scope, reselection="weekly")
    )
    legacy = SelectionState(data, config)

    class DiskState(SelectionState):
        count = 0

        def rank_traders(self, at, effective, scope):
            rows = super().rank_traders(at, effective, scope)
            self.count += 1
            return disk_result(rows, tmp_path / f"source-{self.count}.parquet")

    actual = DiskState(data, config)
    with RankingSink(tmp_path / "actual.parquet") as sink:
        actual.rankings = sink
        for at in (day(config.start), day("2026-01-04")):
            assert actual.advance(at) == legacy.advance(at)
            assert actual.trader_cohorts == legacy.trader_cohorts
            assert actual.previous == legacy.previous
            assert {k: clean(v) for k, v in actual.selected.items()} == {
                k: clean(v) for k, v in legacy.selected.items()
            }
    with RankingSink(tmp_path / "expected.parquet") as expected:
        expected.extend(legacy.rankings)
    assert pq.read_table(sink.artifact.path).equals(
        pq.read_table(expected.artifact.path)
    )


@pytest.fixture
def cohort(tmp_path):
    data, config = dataset()
    state = SelectionState(data, config)
    state.advance(day(config.start))
    scope = next(iter(state.selected))
    rows = [r for r in state.rankings if r["coin"] == scope]
    return (
        disk_result(rows, tmp_path / "source.parquet"),
        rows,
        day(config.start),
        scope,
    )


def append(result, sink, at, scope):
    from arblab.hyperliquid_copy.selection_disk_evidence import append_scored_cohort

    return append_scored_cohort(
        result, sink, at, scope, None, market_time=at, trigger="scheduled"
    )


def test_disk_result_requires_disk_sink(cohort):
    result, _, at, scope = cohort
    with pytest.raises(ValueError, match="RankingSink"):
        append(result, [], at, scope)


@pytest.mark.parametrize(
    "change", ["decision", "scope", "count", "eligible", "selected"]
)
def test_mismatched_result_never_finalizes_sink(tmp_path, cohort, change):
    result, _, at, scope = cohort
    if change == "decision":
        at += timedelta(days=1)
    elif change == "scope":
        scope = "wrong:MARKET"
    elif change == "count":
        result = replace(result, candidate_count=result.candidate_count + 1)
    elif change == "eligible":
        result = replace(result, eligible_count=result.eligible_count + 1)
    else:
        result = replace(result, _selected_json=())
    with pytest.raises(ValueError):
        with RankingSink(tmp_path / "output.parquet") as sink:
            append(result, sink, at, scope)
    with pytest.raises(ValueError, match="not successfully finalized"):
        sink.artifact


def test_large_evidence_bounded_batches_preserves_exclusions(tmp_path, cohort):
    original, rows, at, scope = cohort
    template = rows[0] | dict(
        eligible=False,
        selected=False,
        rank=None,
        score=None,
        weight=0.0,
        exclusions=["no_activity_in_lookback"],
        reasons=["no_activity_in_lookback"],
        percentiles={},
    )
    records = [template | {"user": f"0x{i:040x}"} for i in range(100001)]
    result = disk_result(records, tmp_path / "large.parquet")
    sizes = []

    class BoundedSink(RankingSink):
        def extend(self, batch):
            sizes.append(len(batch))
            assert len(batch) <= 4096
            super().extend(batch)

    with BoundedSink(tmp_path / "large-output.parquet") as sink:
        snapshot, selected = append(result, sink, at, scope)
    assert sink.artifact.rows == snapshot["candidate_count"] == 100001
    assert snapshot["eligible_count"] == snapshot["selected_count"] == 0
    assert selected == [] and sum(sizes) == 100001
    result.artifact.verify()
    assert pq.read_table(sink.artifact.path, columns=["exclusions"])["exclusions"][
        100000
    ].as_py() == ["no_activity_in_lookback"]


@pytest.mark.parametrize("change", ["same", "decision", "scope"])
def test_unbound_empty_artifact_rejects_until_context_contract_exists(
    tmp_path, cohort, change
):
    from arblab.hyperliquid_copy.selection_disk_evidence import append_scored_cohort

    _, _, at, scope = cohort
    result = disk_result([], tmp_path / "empty.parquet")
    if change == "decision":
        at += timedelta(days=1)
    if change == "scope":
        scope = "wrong:MARKET"
    with pytest.raises(ValueError, match="empty.*context"):
        with RankingSink(tmp_path / "empty-output.parquet") as sink:
            append_scored_cohort(
                result, sink, at, scope, ["old"], market_time=at, trigger="scheduled"
            )


def test_source_mutation_during_sink_write_rejects(tmp_path, cohort):
    result, _, at, scope = cohort

    class MutatingSink(RankingSink):
        def extend(self, batch):
            super().extend(batch)
            with result.artifact.path.open("ab") as source:
                source.write(b"changed")

    with pytest.raises(ValueError, match="identity changed"):
        with MutatingSink(tmp_path / "changed-output.parquet") as sink:
            append(result, sink, at, scope)


def test_sink_failure_closes_and_verifies_source(tmp_path, cohort, monkeypatch):
    result, _, at, scope = cohort
    calls = []
    original = type(result.artifact).verify

    def verify(artifact):
        if artifact == result.artifact:
            calls.append(artifact)
        return original(artifact)

    monkeypatch.setattr(type(result.artifact), "verify", verify)

    class FailingSink(RankingSink):
        def extend(self, batch):
            raise OSError("sink interrupted")

    with pytest.raises(OSError, match="sink interrupted"):
        with FailingSink(tmp_path / "failed-output.parquet") as sink:
            append(result, sink, at, scope)
    assert len(calls) >= 2


def test_selected_bound_rejects_before_decoding(tmp_path, cohort):
    result, _, at, scope = cohort
    result = replace(result, _selected_json=("invalid json",) * 251)
    with pytest.raises(ValueError, match="counts"):
        with RankingSink(tmp_path / "oversized.parquet") as sink:
            append(result, sink, at, scope)


def test_failed_scope_does_not_commit_trader_membership(tmp_path):
    data, config = dataset()

    class InvalidState(SelectionState):
        def rank_traders(self, at, effective, scope):
            rows = super().rank_traders(at, effective, scope)
            result = disk_result(rows, tmp_path / "invalid-source.parquet")
            return replace(result, candidate_count=result.candidate_count + 1)

    state = InvalidState(data, config)
    with pytest.raises(ValueError, match="count mismatch"):
        with RankingSink(tmp_path / "invalid-output.parquet") as sink:
            state.rankings = sink
            state.advance(day(config.start))
    assert state.trader_cohorts == [] and state.previous == {} and state.selected == {}
