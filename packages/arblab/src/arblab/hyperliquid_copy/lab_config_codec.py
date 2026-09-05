"""Version dispatch and explicit draft migration; stored v1 payloads stay intact."""

from dataclasses import fields
from .lab_config import LabConfig
from .lab_config_v2 import (
    LabConfigV2,
    TraderSettings,
    FollowerSettings,
    ExplicitUniverse,
)


def parse_lab_config(value):
    if isinstance(value, (LabConfig, LabConfigV2)):
        return value
    if not isinstance(value, dict):
        raise ValueError("Configuration must be an object")
    if (
        value.get("schema_version", "hyperliquid_copy_lab_v1")
        == "hyperliquid_copy_lab_v1"
    ):
        return LabConfig.from_dict(value)
    return LabConfigV2(**value)


def migrate_v1_to_v2(config):
    if isinstance(config, LabConfigV2):
        return config
    return LabConfigV2(
        market_universe=ExplicitUniverse(
            instrument_ids=config.coins,
            allocation="custom",
            weights=config.asset_weights,
        ),
        trader=TraderSettings(
            **{f.name: getattr(config, f.name) for f in fields(TraderSettings)}
        ),
        follower=FollowerSettings(
            **{f.name: getattr(config, f.name) for f in fields(FollowerSettings)}
        ),
        start=config.start,
        end=config.end,
        benchmark=config.benchmark,
        split=config.split,
    )
