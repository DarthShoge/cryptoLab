from datetime import datetime, timezone
import json

import pytest


START = datetime(2026, 8, 3, tzinfo=timezone.utc)
END = datetime(2026, 8, 3, 2, tzinfo=timezone.utc)
MS = int(START.timestamp() * 1000)


def row(offset=80, rate="0.00001", coin="xyz:GOLD"):
    return dict(coin=coin, time=MS + offset, fundingRate=rate, premium="0.0001")


def test_pagination_preserves_actual_settlement_and_hour_bucket():
    from arblab.hyperliquid_copy.proxy_funding import funding_history

    requests = []

    def fetch(payload):
        requests.append(payload)
        return (
            [row()]
            if len(requests) == 1
            else [row(3600048)]
            if len(requests) == 2
            else []
        )

    result = funding_history("xyz:GOLD", START, END, fetch)
    assert len(result.events) == 2
    assert result.events[0].time.microsecond == 80000
    assert result.events[0].hour == START
    assert requests[1]["startTime"] == MS + 81
    assert requests[0]["endTime"] == int(END.timestamp() * 1000) - 1
    assert result.missing_hours == ()


def test_missing_hour_is_evidence_not_zero_rate():
    from arblab.hyperliquid_copy.proxy_funding import funding_history

    pages = iter([[row()], []])
    result = funding_history("xyz:GOLD", START, END, lambda _: next(pages))
    assert len(result.events) == 1
    assert result.missing_hours == (START.replace(hour=1),)


@pytest.mark.parametrize(
    "bad",
    [
        row(coin="BTC"),
        row(rate="nan"),
        row(offset=7200000),
        row(offset=-1),
        row(offset=True),
    ],
)
def test_wrong_coin_invalid_rate_and_out_of_range_rejected(bad):
    from arblab.hyperliquid_copy.proxy_funding import funding_history

    if bad["time"] == MS + 1:
        bad["time"] = True
    with pytest.raises(ValueError):
        funding_history("xyz:GOLD", START, END, lambda _: [bad])


def test_nonadvancing_page_rejected():
    from arblab.hyperliquid_copy.proxy_funding import funding_history

    with pytest.raises(ValueError, match="advance"):
        funding_history("xyz:GOLD", START, END, lambda _: [row()])


def test_multiple_settlements_in_hour_rejected():
    from arblab.hyperliquid_copy.proxy_funding import funding_history

    with pytest.raises(ValueError, match="hour"):
        funding_history("xyz:GOLD", START, END, lambda _: [row(), row(100)])


def test_exact_duplicate_is_deduplicated_but_conflict_is_rejected():
    from arblab.hyperliquid_copy.proxy_funding import funding_history

    pages = iter([[row(), row(), row(3600048)], []])
    assert (
        len(funding_history("xyz:GOLD", START, END, lambda _: next(pages)).events) == 2
    )
    with pytest.raises(ValueError, match="conflict"):
        funding_history("xyz:GOLD", START, END, lambda _: [row(), row(rate="0.0002")])


def test_acquisition_saves_raw_and_preserves_settlement_times(tmp_path):
    from arblab.hyperliquid_copy.proxy_funding_download import download_funding

    pages = iter([[row(), row(3600048)], []])
    path = download_funding(
        ["xyz:GOLD"],
        START,
        END,
        tmp_path,
        fetch=lambda _: json.dumps(next(pages)).encode(),
    )
    manifest = json.loads(path.read_text())
    assert manifest["complete"] is True
    assert manifest["coverage"][0]["events"] == 2
    assert len(manifest["sources"]) == 2
    assert (path.parent / "funding.parquet").exists()
    assert (
        json.loads((path.parent / manifest["sources"][0]["file"]).read_bytes())[0][
            "time"
        ]
        == MS + 80
    )


def test_acquisition_missing_funding_cannot_publish_success(tmp_path):
    from arblab.hyperliquid_copy.proxy_funding_download import download_funding

    with pytest.raises(ValueError, match="Missing"):
        download_funding(["xyz:GOLD"], START, END, tmp_path, fetch=lambda _: b"[]")
    assert not list(tmp_path.glob("*/manifest.json"))
    assert list(tmp_path.glob("*/coverage_failure.json"))
