from datetime import datetime, timezone
from decimal import Decimal

import pytest


def test_canonical_bytes():
    from arblab.hyperliquid_copy.contracts import canonical_json, semantic_hash

    value = {"z": -0.0, "n": None, "a": Decimal("1.2300"), "t": datetime(2026, 1, 1, tzinfo=timezone.utc)}
    assert canonical_json(value) == b'{"a":"1.23","n":null,"t":"2026-01-01T00:00:00.000000Z","z":"0"}'
    assert semantic_hash(value) == semantic_hash(dict(reversed(list(value.items()))))
    for bad in (float("nan"), float("inf"), datetime(2026, 1, 1)):
        with pytest.raises(ValueError):
            canonical_json({"bad": bad})


def test_identifiers_and_paths(tmp_path, monkeypatch):
    from arblab.hyperliquid_copy.contracts import address, symbol
    from arblab.hyperliquid_copy.data_paths import fill_partition, market_partition
    from arblab.paths import hyperliquid_cache_dir

    assert address("0x" + "AB" * 20) == "0x" + "ab" * 20
    for bad in ("0x123", "../bad", "", "0x" + "z" * 40):
        with pytest.raises(ValueError):
            address(bad)
    assert symbol("kPEPE") == "kPEPE"
    for bad in ("../BTC", "xyz:TSLA", "@123", ""):
        with pytest.raises(ValueError):
            symbol(bad)
    monkeypatch.setenv("CRYPTOLAB_ROOT", str(tmp_path))
    assert hyperliquid_cache_dir() == tmp_path / ".hyperliquid_cache"
    assert fill_partition(tmp_path, "2026-01-01").is_relative_to(tmp_path)
    assert market_partition(tmp_path, "BTC", "2026-01-01").name == "l2book.parquet"
    assert list(tmp_path.iterdir()) == []
