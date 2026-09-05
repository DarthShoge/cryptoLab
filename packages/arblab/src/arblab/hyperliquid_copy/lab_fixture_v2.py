"""Fabricated cross-class markets, not real equity/commodity contract definitions."""

from dataclasses import asdict, replace
from datetime import timedelta
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from .lab_fixture import fixture_rows as legacy_rows
from .lab_config import day
from .lab_config_codec import migrate_v1_to_v2
from .lab_config_v2 import LiquidityUniverse
from .download import file_hash

INSTRUMENTS = [
    ("BTC", "crypto", "BTC"),
    ("ETH", "crypto", "ETH"),
    ("SOL", "crypto", "SOL"),
    ("demo:GOLD", "commodity", "ETH"),
    ("demo:STOCK", "equity", "ETH"),
    ("other:STOCK", "equity", "SOL"),
    ("demo:INDEX", "index", "BTC"),
]


def fixture_rows():
    old_fills, old_books, old_funding, old_config, old_metadata = legacy_rows()
    fills = []
    books = []
    funding = []
    instruments = []
    volume = []
    for index, (identifier, asset_class, source) in enumerate(INSTRUMENTS):
        fills.extend(
            replace(f, coin=identifier, event_id=identifier + ":" + f.event_id)
            for f in old_fills
            if f.coin == source
        )
        books.extend(r | {"coin": identifier} for r in old_books if r["coin"] == source)
        funding.extend(
            r | {"coin": identifier} for r in old_funding if r["coin"] == source
        )
        instruments.append(
            dict(
                instrument_id=identifier,
                display_name=identifier.split(":")[-1] + " (synthetic)",
                venue=identifier.split(":")[0] if ":" in identifier else "core",
                asset_class=asset_class,
                base=identifier,
                quote="USD",
                settlement="USDC",
                multiplier=1.0,
                model="linear_usd_continuous_v1",
                known_at=day("2025-12-30"),
                effective_from=day("2025-12-30"),
                effective_to=None,
                listed_at=day("2026-01-01"),
                delisted_at=None,
            )
        )
        for d in range(4):
            at = day("2026-01-01") + timedelta(days=d)
            volume.append(
                dict(
                    instrument_id=identifier,
                    interval_start=at,
                    interval_end=at + timedelta(days=1),
                    available_at=at + timedelta(days=1, seconds=5),
                    notional_usd=float(
                        (index + 1 if d % 2 == 0 else 7 - index) * 1000000
                    ),
                )
            )
    config = replace(
        migrate_v1_to_v2(old_config),
        market_universe=LiquidityUniverse(top_n=3, lookback_days=1),
    )
    metadata = old_metadata | dict(
        schema="hyperliquid_lab_dataset_v2",
        name="Synthetic cross-class market rotation",
        coins=[i[0] for i in INSTRUMENTS],
        snapshot_at="2026-01-06T00:00:00+00:00",
        volume_provenance={
            "source": "fabricated-independent-market-buckets",
            "currency": "USD",
            "conversion": "synthetic USD numeraire",
            "counting": "market_once",
            "interval": "utc_day",
        },
        coverage_note="Fabricated crypto, commodity, equity and index perpetuals; continuous USD-linear model, not real contract qualification",
        default_config=config.to_dict(),
    )
    return fills, books, funding, instruments, volume, config, metadata


def write_fixture(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    fills, books, funding, instruments, volume, _, metadata = fixture_rows()
    files = []
    for name, rows in [
        ("fills.parquet", [asdict(f) for f in fills]),
        ("books.parquet", books),
        ("funding.parquet", funding),
        ("instruments.parquet", instruments),
        ("market_volume.parquet", volume),
    ]:
        path = destination / name
        pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
        files.append(dict(name=name, sha256=file_hash(path), rows=len(rows)))
    metadata["catalogue_hash"] = next(
        f["sha256"] for f in files if f["name"] == "instruments.parquet"
    )
    (destination / "manifest.json").write_text(
        json.dumps(metadata | {"files": files}, indent=2, allow_nan=False)
    )
    return destination
