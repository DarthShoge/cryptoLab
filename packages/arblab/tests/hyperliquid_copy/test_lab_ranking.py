from dataclasses import replace
from datetime import timedelta

from .test_ranking import history


def settings(**changes):
    from arblab.hyperliquid_copy.lab_config import LabConfig

    return LabConfig.from_dict(
        dict(
            coins=["BTC"],
            asset_weights={"BTC": 1},
            min_active_days=1,
            min_episodes=1,
            min_notional=0,
            min_minutes=0,
            min_cohort=1,
            metric_weights={"pnl_efficiency": 1},
            selection="n",
            top_n=2,
            top_fraction=None,
        )
        | changes
    )


def test_top_n_and_fraction_use_eligible_denominator_and_reasons():
    from arblab.hyperliquid_copy.lab_ranking import rank_universe

    fills = history()
    at = max(f.exchange_time for f in fills) + timedelta(seconds=1)
    rows = rank_universe(fills, at, settings(), "BTC", "gross_excludes_fee")
    assert [r["user"] for r in rows if r["selected"]] == [
        "0x" + f"{i:040x}" for i in (5, 4)
    ]
    assert rows[2]["reasons"] == ["rank_below_cutoff"]
    assert rows[0]["percentiles"]["pnl_efficiency"] == 1
    assert sum(r["weight"] for r in rows) == 1
    fractional = rank_universe(
        fills,
        at,
        settings(selection="fraction", top_n=None, top_fraction=0.5),
        "BTC",
        "gross_excludes_fee",
    )
    assert sum(r["selected"] for r in fractional) == 3
    small = rank_universe(
        fills, at, settings(min_cohort=10, max_cohort=10), "BTC", "gross_excludes_fee"
    )
    assert not any(r["selected"] for r in small)


def test_ranking_scope_direction_and_future_causality():
    from arblab.hyperliquid_copy.lab_ranking import rank_universe

    fills = history()
    at = max(f.exchange_time for f in fills) + timedelta(seconds=1)
    config = settings(
        metric_weights={"gross_volume": 1}, metric_directions={"gross_volume": "asc"}
    )
    rows = rank_universe(fills, at, config, "BTC", "gross_excludes_fee")
    assert rows[0]["user"] == "0x" + "0" * 40
    future = [
        replace(f, exchange_time=at + timedelta(days=1), event_id=f.event_id + "future")
        for f in fills
    ]
    assert rows == rank_universe(
        fills + future, at, config, "BTC", "gross_excludes_fee"
    )
    assert rank_universe(fills, at, config, "ETH", "gross_excludes_fee") == []


def test_dormant_wallets_remain_excluded_and_undefined_efficiency_is_missing():
    from arblab.hyperliquid_copy.lab_ranking import rank_universe

    fills = history()
    at = max(f.exchange_time for f in fills) + timedelta(days=2)
    rows = rank_universe(
        fills, at, settings(lookback_days=1), "BTC", "gross_excludes_fee"
    )
    assert len(rows) == 6
    assert all(
        not r["eligible"] and "no_activity_in_lookback" in r["reasons"] for r in rows
    )
    opening = fills[0]
    rows = rank_universe(
        [opening],
        opening.exchange_time + timedelta(seconds=1),
        settings(min_episodes=0),
        "BTC",
        "gross_excludes_fee",
    )
    assert rows[0]["metrics"]["pnl_efficiency"] is None
    assert "missing_ranking_metric" in rows[0]["reasons"]


def test_ranking_output_ceiling_precedes_allocation():
    import pytest
    from arblab.hyperliquid_copy.lab_ranking import validate_ranking_bound

    with pytest.raises(ValueError, match="ranking"):
        validate_ranking_bound(100_000, settings(start="2026-01-01", end="2026-02-01"))
