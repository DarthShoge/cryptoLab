from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from .test_proxy_archive_import import archive
from .test_proxy_download import mapping
from arblab.hyperliquid_copy.download import file_hash
from arblab.hyperliquid_copy.proxy_archive_import import import_archive
from arblab.hyperliquid_copy.proxy_dataset import ProxyDatasetManifest
from arblab.hyperliquid_copy.proxy_registration import register_dataset


def inputs(tmp_path, days=3, start=datetime(2026, 8, 1, tzinfo=timezone.utc)):
    activity = import_archive(
        archive(tmp_path, days=days, start=start), ["BTC"], tmp_path / "activity"
    )
    prices = tmp_path / "prices"
    prices.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                dict(
                    instrument_id="BTC",
                    start=start + timedelta(hours=h),
                    end=start + timedelta(hours=h + 1),
                    open=100.0,
                    high=101.0,
                    low=99.0,
                    close=100.0,
                )
                for h in range(days * 24)
            ]
        ),
        prices / "bars.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                dict(
                    instrument_id="BTC",
                    time=start + timedelta(hours=h),
                    hour=start + timedelta(hours=h),
                    rate=0.0,
                    premium=0.0,
                )
                for h in range(days * 24)
            ]
        ),
        prices / "funding.parquet",
    )
    (prices / "source.raw").write_bytes(b"recorded fixture")
    sources = [
        dict(file="source.raw", bytes=16, sha256=file_hash(prices / "source.raw"))
    ]
    common = dict(
        start=start.isoformat(),
        end=(start + timedelta(days=days)).isoformat(),
        complete=True,
        sources=sources,
    )
    (prices / "prices.json").write_text(
        json.dumps(
            common
            | dict(
                schema="hyperliquid_proxy_prices_v1",
                mappings=[
                    asdict(
                        replace(
                            mapping(),
                            valid_from=start.date().isoformat(),
                            valid_to=(start + timedelta(days=days)).date().isoformat(),
                        )
                    )
                ],
                bars_sha256=file_hash(prices / "bars.parquet"),
                price_policy=dict(
                    adjustment="raw",
                    corporate_actions="none_detected",
                    rolls="not_applicable",
                ),
            )
        )
    )
    (prices / "funding.json").write_text(
        json.dumps(
            common
            | dict(
                schema="hyperliquid_proxy_funding_v1",
                funding_sha256=file_hash(prices / "funding.parquet"),
            )
        )
    )
    return activity, prices / "prices.json", prices / "funding.json"


def test_handoff_week_registers_with_both_hour_eight_sources(tmp_path):
    args = inputs(tmp_path, days=7, start=datetime(2025, 7, 24, tzinfo=timezone.utc))
    target = tmp_path / "lab" / "datasets" / "handoff"
    register_dataset(
        *args, target, start="2025-07-25", end="2025-07-30", name="Handoff fixture"
    )
    manifest = ProxyDatasetManifest(target)
    keys = manifest.metadata["activity_provenance"]["source_keys"]
    assert len(keys) == 121
    assert "node_fills/hourly/20250727/8.lz4" in keys
    assert "node_fills_by_block/hourly/20250727/8.lz4" in keys
    with manifest.load(temp_root=tmp_path, expected_hash=manifest.identity) as loaded:
        assert loaded.activity.count == 240


def test_registers_only_padded_interior_and_loads_native_rows(tmp_path):
    args = inputs(tmp_path)
    target = tmp_path / "lab" / "datasets" / "real"
    register_dataset(
        *args, target, start="2026-08-02", end="2026-08-03", name="Real fixture"
    )
    manifest = ProxyDatasetManifest(target)
    assert not manifest.metadata["synthetic"]
    assert len(manifest.metadata["activity_provenance"]["source_keys"]) == 24
    assert len(manifest.metadata["activity_provenance"]["padding_source_keys"]) == 48
    with manifest.load(temp_root=tmp_path, expected_hash=manifest.identity) as loaded:
        assert loaded.activity.count == 48


def test_refuses_unpadded_source_end_before_registering(tmp_path):
    args = inputs(tmp_path)
    target = tmp_path / "real"
    with pytest.raises(ValueError, match="padding"):
        register_dataset(
            *args, target, start="2026-08-02", end="2026-08-04", name="No padding"
        )
    assert not target.exists()


def test_source_corruption_cannot_be_registered(tmp_path):
    args = inputs(tmp_path)
    (args[1].parent / "source.raw").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity"):
        register_dataset(
            *args,
            tmp_path / "real",
            start="2026-08-02",
            end="2026-08-03",
            name="Bad source",
        )
