import hashlib
import io
import json
import zipfile
from dataclasses import asdict, replace
from datetime import datetime, timezone, timedelta

import pytest


def mapping():
    from arblab.hyperliquid_copy.proxy_mapping import ProxyMapping

    return ProxyMapping(
        "BTC",
        "binance",
        "BTCUSDT",
        "crypto",
        "USD",
        "24/7",
        "BTC",
        "raw",
        "2026-08-01",
        "2026-09-01",
        "USD-M price proxy, USDT assumed USD",
        "Binance public archive",
    )


def archive():
    out = io.BytesIO()
    rows = "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n"
    for hour in range(24):
        ms = 1785715200000 + hour * 3600000
        rows += f"{ms},100,110,90,105,1,{ms + 3599999},100,1,1,100,0\n"
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("BTCUSDT-1h-2026-08-03.csv", rows)
    return out.getvalue()


def dates():
    return datetime(2026, 8, 3, tzinfo=timezone.utc), datetime(
        2026, 8, 4, tzinfo=timezone.utc
    )


def test_download_preserves_raw_checksums_and_normalized_coverage(tmp_path):
    from arblab.hyperliquid_copy.proxy_download import download_prices

    raw = archive()

    def fetch(url):
        return (
            (hashlib.sha256(raw).hexdigest() + "  file.zip").encode()
            if url.endswith("CHECKSUM")
            else raw
        )

    path = download_prices([mapping()], *dates(), tmp_path, fetch=fetch)
    manifest = json.loads(path.read_text())
    assert manifest["mode"] == "approximate_proxy_prices"
    assert manifest["mappings"] == [asdict(mapping())]
    assert manifest["coverage"][0]["bars"] == 24
    assert manifest["coverage"][0]["missing_starts"] == []
    assert manifest["price_policy"] == dict(
        adjustment="raw", corporate_actions="none_detected", rolls="not_applicable"
    )
    assert manifest["sources"][0]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert (path.parent / "bars.parquet").exists()
    assert (path.parent / manifest["sources"][0]["file"]).read_bytes() == raw


def test_bad_checksum_does_not_publish_success_manifest(tmp_path):
    from arblab.hyperliquid_copy.proxy_download import download_prices

    with pytest.raises(ValueError, match="checksum"):
        download_prices(
            [mapping()],
            *dates(),
            tmp_path,
            fetch=lambda url: b"0" * 64 if url.endswith("CHECKSUM") else archive(),
        )
    assert not list(tmp_path.glob("*/manifest.json"))


def test_mapping_must_cover_whole_requested_range(tmp_path):
    from arblab.hyperliquid_copy.proxy_download import download_prices

    start, end = dates()
    with pytest.raises(ValueError, match="mapping"):
        download_prices(
            [mapping()],
            start.replace(month=7),
            end,
            tmp_path,
            fetch=lambda url: pytest.fail("must validate before network"),
        )


def test_future_coverage_rejected_before_fetch(tmp_path):
    from arblab.hyperliquid_copy.proxy_download import download_prices

    future = replace(mapping(), valid_from="2100-08-01", valid_to="2100-09-01")
    start, end = (d.replace(year=2100) for d in dates())
    with pytest.raises(ValueError, match="future"):
        download_prices(
            [future],
            start,
            end,
            tmp_path,
            fetch=lambda url: pytest.fail("must not fetch future bars"),
        )


def test_closed_equity_session_does_not_make_crypto_coverage_incomplete(tmp_path):
    from arblab.hyperliquid_copy.proxy_download import download_prices

    stock = replace(
        mapping(),
        instrument_id="xyz:TSLA",
        provider="yahoo",
        ticker="TSLA",
        asset_class="equities",
        calendar="XNYS",
    )
    start, end = (d + timedelta(days=5) for d in dates())
    original = zipfile.ZipFile(io.BytesIO(archive()))
    lines = original.read(original.namelist()[0]).decode().splitlines()
    converted = [lines[0]]
    for line in lines[1:]:
        row = line.split(",")
        row[0], row[6] = (
            str(int(row[0]) + 5 * 86400000),
            str(int(row[6]) + 5 * 86400000),
        )
        converted.append(",".join(row))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("BTCUSDT-1h-2026-08-08.csv", "\n".join(converted))
    raw = out.getvalue()

    def fetch(url):
        if "yahoo" in url:
            return json.dumps(
                {
                    "chart": {
                        "result": [
                            {
                                "meta": {"currency": "USD", "instrumentType": "EQUITY"},
                                "timestamp": [],
                                "indicators": {
                                    "quote": [dict(open=[], high=[], low=[], close=[])]
                                },
                            }
                        ]
                    }
                }
            ).encode()
        return (
            hashlib.sha256(raw).hexdigest().encode()
            if url.endswith("CHECKSUM")
            else raw
        )

    path = download_prices([mapping(), stock], start, end, tmp_path, fetch=fetch)
    manifest = json.loads(path.read_text())
    assert manifest["complete"] is True
    assert manifest["coverage"][1]["scheduled_bars"] == 0
    assert manifest["coverage"][1]["session_closed_entire_range"] is True
