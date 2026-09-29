from datetime import datetime, timedelta, timezone

import pytest


def test_derived_metrics_and_suppression():
    from hyperliquid_explorer_api.analytics import build_metrics
    from hyperliquid_explorer_api.models import Scenario

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    stats = dict(
        start=start,
        end=start + timedelta(days=2),
        samples=2881,
        initial=100,
        final=110,
        current_drawdown=0.02,
        maximum_drawdown=0.1,
        mean_gross_leverage=0.5,
        mean_net_leverage=-0.1,
        maximum_gross_leverage=0.7,
        invalid_equity=0,
        gaps=0,
    )
    scenario = Scenario(
        scenario_type="strategy",
        name="x",
        latency_seconds=5,
        metrics=dict(
            sharpe=1.2,
            sortino=2,
            max_drawdown=0.1,
            annualized_return=0.5,
            fee_drag=0.01,
            funding_drag=-0.02,
        ),
    )
    values = build_metrics(scenario, stats, False)
    assert values["calmar"].value == 5
    assert values["net_pnl"].value == 10
    assert values["fees_usd"].value == 1
    assert values["funding_usd"].value == -2
    assert values["mean_net_leverage"].value == -0.1
    assert values["sharpe"].value == 1.2
    assert build_metrics(scenario, stats, True)["sharpe"].reason == "synthetic demo"
    assert (
        build_metrics(scenario, stats | {"end": start + timedelta(minutes=3)}, False)[
            "calmar"
        ].reason
        == "insufficient history"
    )
    assert (
        build_metrics(scenario, stats | {"gaps": 1}, False)["mean_gross_leverage"].value
        is None
    )
    scenario.metrics["max_drawdown"] = 0
    assert build_metrics(scenario, stats, False)["calmar"].value is None


def test_curve_downsampling_preserves_drawdown_and_bounded_output(report_root):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from hyperliquid_explorer_api.repository import Repository
    from hyperliquid_explorer_api.queries import Curve

    root, run = report_root
    at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [
        dict(
            time=at + timedelta(minutes=i),
            equity=50.0 if i == 3456 else 100.0,
            cash=100.0,
            unrealized_pnl=0.0,
            gross_exposure=float(i % 17),
            net_exposure=float(i % 7),
            signal_name="direction_equal",
            latency_seconds=5,
        )
        for i in range(10000)
    ]
    pq.write_table(pa.Table.from_pylist(rows), root / run / "equity_curve.parquet")
    repo = Repository(root)
    curve = Curve(repo, run, repo.scenario(run, "strategy", "direction_equal", 5))
    page = curve.page(200)
    assert page.total == 10000 and len(page.rows) <= 200
    assert page.rows[0].time == at and page.rows[-1].time == rows[-1]["time"]
    assert max(r.drawdown for r in page.rows) == 0.5
    assert curve.stats()["maximum_drawdown"] == 0.5


@pytest.mark.parametrize(
    "column",
    ["equity", "gross_exposure", "net_exposure", "cash", "unrealized_pnl", "time"],
)
def test_missing_curve_values_are_not_silently_averaged(report_root, column):
    import pyarrow.parquet as pq
    from hyperliquid_explorer_api.analytics import build_metrics
    from hyperliquid_explorer_api.models import Scenario
    from hyperliquid_explorer_api.repository import Repository, ReportError
    from hyperliquid_explorer_api.queries import Curve

    root, run = report_root
    path = root / run / "equity_curve.parquet"
    table = pq.read_table(path)
    values = table[column].to_pylist()
    values[0] = None
    import pyarrow as pa

    table = table.set_column(
        table.schema.get_field_index(column),
        column,
        pa.array(values, type=table[column].type),
    )
    pq.write_table(table, path)
    scenario = Scenario(
        scenario_type="strategy",
        name=table["signal_name"][0].as_py(),
        latency_seconds=table["latency_seconds"][0].as_py(),
        metrics={},
    )
    curve = Curve(Repository(root), run, scenario)
    stats = curve.stats()
    assert stats["invalid_equity"] > 0
    assert build_metrics(scenario, stats, False)["mean_gross_leverage"].value is None
    with pytest.raises(ReportError):
        curve.page()
