"""Explicitly fabricated multi-decision development dataset; never exchange data."""

from dataclasses import asdict
from datetime import timedelta
from math import sin

from .contracts import FillEvent
from .lab_config import LabConfig, day
from .market_data import MarketData


def fixture_rows():
    warmup, start, end = day("2026-01-01"), day("2026-01-03"), day("2026-01-05")
    fills = []
    for coin in ("ETH", "SOL", "BTC"):
        for user_number in range(6):
            user = "0x" + f"{user_number + 1:040x}"
            previous = 0.0
            # Daily completed episodes and overnight reopening. Later winners
            # differ from earlier winners; ETH and SOL favour different wallets.
            for d in range(4):
                at = warmup + timedelta(days=d, hours=1)
                direction = 1 if user_number % 2 == 0 else -1
                profit = (user_number + 1 if d < 2 else 6 - user_number) * (
                    1 if coin != "SOL" else -1
                )
                qty = (user_number + 1) * direction
                events = (
                    [
                        (
                            at,
                            -previous,
                            previous,
                            0.0,
                            profit,
                            "Close Long" if previous > 0 else "Close Short",
                        )
                    ]
                    if previous
                    else []
                )
                events += [
                    (
                        at + timedelta(minutes=1),
                        qty,
                        0.0,
                        qty,
                        0.0,
                        "Open Long" if qty > 0 else "Open Short",
                    ),
                    (
                        at + timedelta(hours=1),
                        -qty,
                        qty,
                        0.0,
                        profit,
                        "Close Long" if qty > 0 else "Close Short",
                    ),
                    (
                        at + timedelta(hours=2),
                        qty,
                        0.0,
                        qty,
                        0.0,
                        "Open Long" if qty > 0 else "Open Short",
                    ),
                ]
                for event_index, (
                    moment,
                    delta,
                    before,
                    after,
                    pnl,
                    direction_name,
                ) in enumerate(events):
                    identifier = f"{coin}:{user_number}:{d}:{event_index}"
                    fills.append(
                        FillEvent(
                            identifier,
                            moment,
                            None,
                            None,
                            len(fills),
                            0,
                            user,
                            coin,
                            100.0,
                            abs(delta),
                            "B" if delta > 0 else "A",
                            before,
                            after,
                            direction_name,
                            float(pnl),
                            0.01,
                            "USDC",
                            True,
                            len(fills),
                            "synthetic",
                            len(fills),
                            False,
                            "synthetic-fixture",
                            end,
                        )
                    )
                previous = qty
    books, funding = [], []
    at = warmup
    while at <= end:
        index = (at - warmup).total_seconds() / 60
        for c, coin in enumerate(("BTC", "ETH", "SOL")):
            price = 100 + (index / 1440) * (c + 1) + sin(index / 300 + c) * 2
            # Actual fixture books, not a fallback interpolator. Supported delay
            # coverage is checked against these observations before each run.
            for seconds in (0, 1, 5, 15):
                if at + timedelta(seconds=seconds) <= end:
                    books.append(
                        dict(
                            exch_time=at + timedelta(seconds=seconds),
                            coin=coin,
                            bids=[dict(px=price - 0.01, sz=10000.0)],
                            asks=[dict(px=price + 0.01, sz=10000.0)],
                        )
                    )
            if at.minute == 0:
                funding.append(
                    dict(
                        timestamp=at,
                        coin=coin,
                        rate=0.00001 if coin != "SOL" else -0.00001,
                    )
                )
        at += timedelta(minutes=1)
    config = LabConfig.from_dict(
        dict(
            lookback_days=1,
            min_active_days=1,
            min_episodes=1,
            min_notional=0,
            min_minutes=0,
            min_cohort=1,
            max_cohort=6,
            min_known=1,
            scale_lookback_days=1,
            selection="n",
            top_n=2,
            top_fraction=None,
            metric_weights={"pnl_efficiency": 1},
            update_minutes=60,
            deadband=0,
            min_trade_usd=0,
            start="2026-01-03",
            end="2026-01-05",
        )
    )
    metadata = dict(
        schema="hyperliquid_lab_dataset_v1",
        name="Synthetic rotating SOL / ETH cohorts",
        synthetic=True,
        coverage_start="2026-01-01",
        coverage_end="2026-01-05",
        coins=["BTC", "ETH", "SOL"],
        fee_semantics="gross_excludes_fee",
        coverage_note="Six fabricated wallets per asset; relaxed eligibility; not an all-wallet market archive",
        default_config=asdict(config),
    )
    return fills, books, funding, config, metadata


def fixture():
    fills, books, funding, config, metadata = fixture_rows()
    return fills, MarketData(books, funding), config, metadata


def write_fixture(destination):
    import json
    from pathlib import Path
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .download import file_hash

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    fills, books, funding, _, metadata = fixture_rows()
    files = []
    for name, rows in (
        ("fills.parquet", [asdict(f) for f in fills]),
        ("books.parquet", books),
        ("funding.parquet", funding),
    ):
        path = destination / name
        pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
        files.append(dict(name=name, sha256=file_hash(path), rows=len(rows)))
    (destination / "manifest.json").write_text(
        json.dumps(metadata | {"files": files}, indent=2, allow_nan=False)
    )
    return destination
