import pytest
from arblab.hyperliquid_copy.lab_config import LabConfig
from arblab.hyperliquid_copy.contracts import symbol, semantic_hash


def test_version_adapter_preserves_v1_and_roundtrips_v2():
    from arblab.hyperliquid_copy.lab_config_codec import (
        parse_lab_config,
        migrate_v1_to_v2,
    )

    old = LabConfig()
    before = semantic_hash(old.to_dict())
    new = migrate_v1_to_v2(old)
    assert new.market_universe.instrument_ids == ["ETH", "SOL"]
    assert new.trader.lookback_days == 90
    assert new.effective(["demo:GOLD"], {"demo:GOLD": 1}).coins == ["demo:GOLD"]
    assert parse_lab_config(new.to_dict()).to_dict() == new.to_dict()
    assert semantic_hash(parse_lab_config(old.to_dict()).to_dict()) == before


def test_general_and_mode_specific_fields_are_exclusive():
    from arblab.hyperliquid_copy.lab_config_v2 import parse_universe

    valid = dict(mode="liquidity", general=True, classes=[], top_n=3)
    assert parse_universe(valid).top_n == 3
    for invalid in [
        valid | {"classes": ["crypto"]},
        valid | {"instrument_ids": ["BTC"]},
        valid | {"top_n": True},
        valid | {"publication_lag_days": 0},
        dict(mode="explicit", general=True, classes=[], instrument_ids=["BTC"]),
        dict(
            mode="explicit",
            classes=["crypto"],
            instrument_ids=["BTC"],
            allocation="custom",
            weights={"BTC": 0.5},
        ),
    ]:
        with pytest.raises((ValueError, TypeError)):
            parse_universe(invalid)


def test_namespaces_are_preserved_but_paths_rejected():
    assert symbol("demo:ABC") == "demo:ABC"
    assert symbol("other:ABC") != symbol("demo:ABC")
    for invalid in ["../ABC", "a:b:c", "demo: A", "a/b", "demo:"]:
        with pytest.raises(ValueError):
            symbol(invalid)
